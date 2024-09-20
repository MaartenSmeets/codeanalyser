import os
import time
import shelve
import uuid
import argparse
import logging
import traceback
import chardet
import shutil
import json
import subprocess
import re
from hashlib import md5
from transformers import AutoTokenizer
import requests
from datetime import datetime
from huggingface_hub import login

# Constants and Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"  # Configurable Ollama URL

DEFAULT_SUMMARIZATION_MODEL = 'gemma2:9b-instruct-q8_0'
DEFAULT_SUMMARIZATION_TOKENIZER_NAME = "google/gemma-2-9b-it"
MAX_SUMMARIZATION_CONTEXT_LENGTH = 4096  # Max token length for summarization model

DEFAULT_PLANTUML_MODEL = 'dolphin-mixtral:8x22b'
DEFAULT_PLANTUML_TOKENIZER_NAME = "cognitivecomputations/dolphin-2.9-mixtral-8x22b"
MAX_PLANTUML_CONTEXT_LENGTH = 8192  # Max token length for PlantUML model

CACHE_DIR = 'llm_cache'
SUMMARIES_DIR = "summaries"
IRRELEVANT_SUMMARIES_DIR = "irrelevant_summaries"
UNPROCESSED_DIR = "unprocessed_files"
PLANTUML_FILE = "codebase_diagram.puml"
PLANTUML_PNG_FILE = "codebase_diagram.png"
PLANTUML_PROMPT_FILE = "plantuml_prompt.txt"

# Logging Configuration
log_file = 'script_run.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
)

# Prompt Templates
DEFAULT_PLANTUML_PROMPT_TEMPLATE = """Based on the following comprehensive codebase summary, generate a valid PlantUML diagram that accurately represents the system's architecture and functional flow. The diagram should be high-level, focusing on the major components, their interactions, and data flow.

Instructions:

- The output should be valid PlantUML code, enclosed within @startuml and @enduml.
- Use appropriate PlantUML elements such as packages, classes, interfaces, and arrows to represent the components and their relationships.
- Focus on the architectural and functional relationships between components.
- Ensure that the syntax is correct and that the diagram can be rendered without errors.
- Do not include any explanations or descriptions outside of the PlantUML code.
- Output only the PlantUML code, starting with '@startuml' and ending with '@enduml', with no additional text or explanations before or after.

Remember, your task is to generate a PlantUML diagram based on the provided codebase summary, not to summarize the context.

**Codebase Summary**:
{combined_summary}
"""

FILE_SUMMARY_PROMPT_TEMPLATE = """You are tasked with summarizing a file from a software repository. Provide a **precise**, **comprehensive**, and **well-structured** English summary that accurately reflects the contents of the file. Do not ask for confirmation. Do not provide suggestions or recommendations. Focus solely on creating the summary and keep it concise. Avoid redundancy and do not summarize the summary. The summary must be:

- **Factual and objective**: Include only verifiable information based on the provided file. Avoid any assumptions, opinions, interpretations, or speculative conclusions.
- **Specific and relevant**: Directly reference the actual contents of the file. Avoid general statements or unrelated information. Focus on the specific purpose, functionality, and structure of the file.
- **Concise yet complete**: Ensure that the summary captures all essential details while being succinct. Eliminate redundancy and unnecessary information.

In particular, address the following when applicable and relevant to the file’s role in the codebase:
- **Purpose and functionality**: Describe the file's core purpose, what functionality it implements, and how it fits into the broader system.
- **Key components**: Highlight any critical functions, classes, methods, or modules defined in the file and explain their roles.
- **Inputs and outputs**: Explicitly mention any input data or parameters the file processes, and describe the outputs it generates.
- **Dependencies**: Identify any internal or external dependencies (e.g., libraries, APIs, other files) and explain how they are used in the file.
- **Data flow**: Describe the flow of data through the file, including how data is processed, transformed, or manipulated.
- **Interactions**: If applicable, detail how this file interacts with other parts of the system or external systems.

Your summary should provide enough detail to give a clear understanding of the file’s purpose and its function within the codebase, without adding unnecessary explanations or speculative content.

**File being summarized**: {file_path}
{file_content}
"""

CHUNK_SUMMARY_PROMPT_TEMPLATE = """You are summarizing a portion of a file in a software repository. This portion belongs to a larger file, and it is part {chunk_index} of {total_chunks}. Summarize strictly and solely the content of this chunk in a clear, structured, and concise manner, focusing on relevant technical and functional details. Do not provide recommendations or ask follow-up questions.

Key instructions for summarizing:
- **Do not make any assumptions** about other chunks or the overall file context.
- **Be factual**, specific, and avoid redundancy.
- Only include information that is clearly evident within this chunk, such as:
  - **Inputs** and **outputs** of functions or components.
  - **Dependencies** (e.g., external libraries, APIs, or other files).
  - **Data flow**, how data is transformed or processed within this chunk.
  - **Key functions, classes, methods**, and their purposes.
- Exclude any assumptions, opinions, or evaluations.

The goal is to accurately capture the functionality and purpose of this specific chunk within the file.
- **Filename**: {file_path}

**Chunk content**:
{chunk_content}
"""

CHUNK_SUMMARIES_CONSOLIDATION_PROMPT = """You have summarized a large file in multiple chunks. Now, based on the following individual chunk summaries, create a single cohesive, structured, and factual summary of the entire file. Strictly and only provide the summary. Do not summarize the summary. Do not ask for confirmation. Do not provide suggestions. Do not suggest or ask additional input or follow-up questions. Do not indicate potential uses or interpretations. Do not provide recommendations.

File: {file_path}

Instructions for the final summary:
- **Integrate** all the relevant information from the chunk summaries without duplicating content.
- Provide a clear **overview of the file's purpose** and its role within the software repository.
- Highlight the most significant **functions, classes, modules**, or **methods**, and describe their specific roles in the file.
- Mention any **dependencies** (e.g., external libraries, APIs) and how they interact with the file.
- Include key information about **inputs**, **outputs**, and **data flow**.
- Avoid any **assumptions**, **opinions**, **irrelevant details** or **asking/suggesting to the user** (such as 'let me know'.. consider the user non-existent).
- Ensure the summary is concise yet comprehensive enough to convey the file’s overall functionality.

Here are the chunk summaries:
{chunk_summaries}
"""

EVALUATE_RELEVANCE_PROMPT = """Based on the following summary, determine whether the file is relevant for generating an architectural diagram of the codebase. If the file is central to the architecture or contains significant components that should be represented in the diagram, respond with 'Yes'. If the file is not relevant, respond with 'No'.

Summary:
{summary}

Is this file relevant for generating the architectural diagram? Respond with 'Yes' or 'No' only.
"""

# Helper Functions
def init_cache() -> shelve.Shelf:
    """Initialize the shelve cache."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return shelve.open(os.path.join(CACHE_DIR, 'llm_cache.db'))

def clean_generated_summary(summary: str) -> str:
    """Clean and format the final summary."""
    cleaned_summary = "\n".join(
        [sentence for sentence in summary.split("\n") if not sentence.startswith("Let me know")]
    )
    return cleaned_summary.rstrip()

def generate_cache_key(prompt: str, model: str) -> str:
    """Generate a unique hash for cache key based on input."""
    key_string = f"{model}_{prompt}"
    return md5(key_string.encode()).hexdigest()

def get_tokenizer(tokenizer_name: str):
    """Initialize tokenizer based on the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True)
    return tokenizer

def split_into_chunks(text: str, max_tokens: int, tokenizer) -> list:
    """Split text into chunks based on token length."""
    logging.debug(f"Starting tokenization of the text for splitting...")
    tokens = tokenizer(text, return_tensors='pt',
                       add_special_tokens=False).input_ids[0]

    logging.debug(f"Total number of tokens: {len(tokens)}")

    if len(tokens) == 0:
        logging.error("Tokenization returned no tokens. Skipping chunking.")
        return []

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))

    logging.info(f"File split into {len(chunks)} chunks.")
    return chunks

def generate_unique_filename(base_name: str, extension: str) -> str:
    """Generate a unique filename with timestamp and unique ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    return f"{base_name}_{timestamp}_{unique_id}.{extension}"

def generate_response_with_ollama(prompt: str, model: str) -> str:
    """Call the LLM via Ollama to generate responses with caching."""
    cache = init_cache()
    cache_key = generate_cache_key(prompt, model)

    # Check if the result is already cached
    if cache_key in cache:
        logging.info(
            f"Fetching result from cache for prompt: {prompt[:50]}...")
        response_content = cache[cache_key]
        cache.close()
        return response_content

    try:
        logging.debug(
            f"Sending request to Ollama with model '{model}' and prompt size {len(prompt)}")
        url = OLLAMA_URL
        payload = {
            "model": model,
            "prompt": prompt
        }
        headers = {'Content-Type': 'application/json'}

        # Send the request
        response = requests.post(url, data=json.dumps(payload), headers=headers, stream=True)

        if response.status_code != 200:
            logging.error(f"Failed to generate response with Ollama: HTTP {response.status_code}")
            return ""

        response_content = ''
        for line in response.iter_lines():
            if line:
                # Each line is a JSON-formatted string
                try:
                    line_json = json.loads(line.decode('utf-8'))
                    response_text = line_json.get('response', '')
                    response_content += response_text
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse line as JSON: {e}")
                    continue

        if not response_content:
            logging.warning(f"Unexpected response or no response.")
            return ""

        # Cache the result
        cache[cache_key] = response_content
        cache.close()

        return response_content

    except Exception as e:
        logging.error(f"Failed to generate response with Ollama: {e}")
        logging.debug(f"Prompt used: {prompt[:200]}...")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        cache.close()
        raise e

def summarize_chunk_summaries(chunk_summaries: list, file_path: str, summarization_model: str, summarization_tokenizer_name: str) -> str:
    """Summarize chunks and return the result."""
    chunk_summary_text = "\n\n".join(chunk_summaries)

    prompt = CHUNK_SUMMARIES_CONSOLIDATION_PROMPT.format(
        file_path=file_path,
        chunk_summaries=chunk_summary_text
    )

    final_summary = generate_response_with_ollama(prompt, summarization_model)
    return clean_generated_summary(final_summary)

def extract_plantuml_code(content: str) -> str:
    """Extract PlantUML code from LLM output."""
    pattern = re.compile(r'@startuml.*?@enduml', re.DOTALL)
    match = pattern.search(content)
    if match:
        return match.group(0)
    else:
        logging.warning("PlantUML tags @startuml and @enduml not found in the LLM output.")
        return content.strip()

def evaluate_relevance(summary: str, summarization_model: str) -> bool:
    """Evaluate if the summary is relevant for generating the diagram."""
    prompt = EVALUATE_RELEVANCE_PROMPT.format(summary=summary)

    response = generate_response_with_ollama(prompt, summarization_model)
    response_cleaned = response.strip().lower()
    return response_cleaned.startswith('yes')

def summarize_codebase(directory: str, summarization_model: str, summarization_tokenizer_name: str, plantuml_model: str, plantuml_tokenizer_name: str, plantuml_context: str = DEFAULT_PLANTUML_PROMPT_TEMPLATE) -> str:
    """Summarize the entire repository and generate the PlantUML diagram."""
    all_files = list_all_files(directory)
    total_files = len(all_files)
    codebase_summary = []
    logging.info(f"Total files to process: {total_files}")

    if not os.path.exists(SUMMARIES_DIR):
        os.makedirs(SUMMARIES_DIR)

    if not os.path.exists(IRRELEVANT_SUMMARIES_DIR):
        os.makedirs(IRRELEVANT_SUMMARIES_DIR)

    if not os.path.exists(UNPROCESSED_DIR):
        os.makedirs(UNPROCESSED_DIR)

    # Process each file
    for idx, file_path in enumerate(all_files, start=1):
        logging.info(f"Processing file {idx}/{total_files}: {file_path}")
        file_content = read_file(file_path, directory, UNPROCESSED_DIR)

        if not file_content:
            logging.warning(f"Skipping unreadable or empty file: {file_path}")
            continue

        try:
            summary, is_test_file_flag = generate_summary(file_path, file_content, summarization_model, summarization_tokenizer_name)

            if is_test_file_flag:
                logging.info(f"Test or irrelevant file detected and skipped: {file_path}")
                continue

            if summary:
                # Evaluate relevance
                is_relevant = evaluate_relevance(summary, summarization_model)

                # Include relative path and filename in the summary for reference
                cleaned_summary = clean_generated_summary(summary)
                formatted_summary = f"File: {file_path}\n\n{cleaned_summary}\n"

                if is_relevant:
                    codebase_summary.append(formatted_summary)
                    file_summary_path = generate_unique_filename(
                        os.path.basename(file_path), "summary.txt")
                    save_output_to_file(formatted_summary, os.path.join(
                        SUMMARIES_DIR, file_summary_path))
                else:
                    logging.info(f"File '{file_path}' deemed not relevant for the diagram.")
                    file_summary_path = generate_unique_filename(
                        os.path.basename(file_path), "summary.txt")
                    save_output_to_file(formatted_summary, os.path.join(
                        IRRELEVANT_SUMMARIES_DIR, file_summary_path))

            if idx % 5 == 0 or idx == total_files:
                logging.info(f"Progress: {idx}/{total_files} files processed.")

        except Exception as e:
            logging.error(f"Error processing file: {file_path}")
            logging.error(f"Exception details: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

            # Copy the file to unprocessed_files if it fails
            copy_unreadable_file(file_path, directory, UNPROCESSED_DIR)

    # Combine all relevant file summaries for the PlantUML diagram
    combined_summary = "\n".join(codebase_summary)

    if combined_summary:
        # Save the final codebase summary
        logging.info("Final codebase summary generated.")
        summary_file = generate_unique_filename("codebase_summary", "txt")
        save_output_to_file(combined_summary, summary_file)

        # Generate the PlantUML prompt
        llm_plantuml_prompt = plantuml_context.format(combined_summary=combined_summary)

        # Save the PlantUML prompt context to a file for debugging
        save_output_to_file(llm_plantuml_prompt, PLANTUML_PROMPT_FILE)

        logging.info("Preparing to generate PlantUML diagram...")

        # Check if the prompt exceeds the maximum context length
        plantuml_tokenizer = get_tokenizer(plantuml_tokenizer_name)
        prompt_token_count = len(
            plantuml_tokenizer(llm_plantuml_prompt, return_tensors="pt").input_ids[0]
        )

        if prompt_token_count > MAX_PLANTUML_CONTEXT_LENGTH:
            logging.warning("PlantUML prompt exceeds maximum context length. Truncating summaries.")
            # Truncate the combined_summary to fit within MAX_PLANTUML_CONTEXT_LENGTH
            available_tokens = MAX_PLANTUML_CONTEXT_LENGTH - (prompt_token_count - len(plantuml_tokenizer(combined_summary, return_tensors="pt").input_ids[0]))
            truncated_summary = truncate_text_to_token_limit(combined_summary, available_tokens, plantuml_tokenizer)
            llm_plantuml_prompt = plantuml_context.format(combined_summary=truncated_summary)
            # Save the truncated prompt
            save_output_to_file(llm_plantuml_prompt, PLANTUML_PROMPT_FILE)

        logging.info("Restarting Ollama to free up memory before generating PlantUML diagram...")

        # Restart Ollama to free up memory before generating PlantUML diagram
        try:
            subprocess.call("./restart_ollama.sh", shell=True)
            while True:
                try:
                    response = requests.get(OLLAMA_URL, timeout=2)  # A short timeout to ensure responsiveness
                    if response.status_code == 404:
                        logging.info("Ollama service is up and running.")
                        break
                except requests.exceptions.RequestException as e:
                    logging.info("Ollama service not yet available, retrying...")
                time.sleep(5)
            
            logging.info("Successfully restarted Ollama.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to restart Ollama: {e}")
            return ""

        logging.info("Generating PlantUML diagram...")

        # Call LLM to generate a PlantUML diagram structure
        plantuml_diagram_content = generate_response_with_ollama(llm_plantuml_prompt, plantuml_model)

        # Extract valid PlantUML code from the LLM output
        plantuml_code = extract_plantuml_code(plantuml_diagram_content)

        # Save the PlantUML diagram content to a file
        with open(PLANTUML_FILE, 'w') as f:
            f.write(plantuml_code)

        # Convert the PlantUML diagram to PNG using PlantUML
        try:
            subprocess.run(["plantuml", "-tpng", PLANTUML_FILE], check=True)
            logging.info(f"PlantUML diagram saved as PNG.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate PlantUML PNG: {e}")

    return combined_summary

def truncate_text_to_token_limit(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate text to fit within a maximum token limit."""
    tokens = tokenizer(text, return_tensors='pt',
                       add_special_tokens=False).input_ids[0]
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text

def generate_summary(file_path: str, file_content: str, summarization_model: str, summarization_tokenizer_name: str) -> tuple:
    """Generate a summary for each file."""
    if is_test_file(file_path):
        logging.info(f"Skipping test file: {file_path}")
        return None, True

    if len(file_content.strip()) == 0:
        logging.warning(f"Skipping empty file: {file_path}")
        return None, True

    tokenizer = get_tokenizer(summarization_tokenizer_name)

    # Prepare the prompt
    prompt = FILE_SUMMARY_PROMPT_TEMPLATE.format(
        file_path=file_path,
        file_content=file_content
    )

    full_prompt_token_count = len(
        tokenizer(prompt, return_tensors="pt").input_ids[0]
    )

    logging.debug(f"Full prompt token count for file '{file_path}': {full_prompt_token_count}")

    if full_prompt_token_count > MAX_SUMMARIZATION_CONTEXT_LENGTH:
        logging.debug(f"File '{file_path}' exceeds context length; processing in chunks.")
        prompt_token_count = len(
            tokenizer(FILE_SUMMARY_PROMPT_TEMPLATE.format(file_path=file_path, file_content=""), return_tensors="pt").input_ids[0]
        )
        available_tokens_for_content = MAX_SUMMARIZATION_CONTEXT_LENGTH - prompt_token_count

        if available_tokens_for_content <= 0:
            logging.error(f"Not enough space for content in the context for file '{file_path}'. Skipping file.")
            return None, True

        chunks = split_into_chunks(file_content, available_tokens_for_content, tokenizer)
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            chunk_prompt = CHUNK_SUMMARY_PROMPT_TEMPLATE.format(
                chunk_index=i+1,
                total_chunks=len(chunks),
                file_path=file_path,
                chunk_content=chunk
            )

            logging.info(f"Processing chunk {i+1}/{len(chunks)} for file '{file_path}'")

            try:
                chunk_summary = generate_response_with_ollama(chunk_prompt, summarization_model)
                cleaned_chunk_summary = clean_generated_summary(chunk_summary)
                chunk_filename = generate_unique_filename(f"{os.path.basename(file_path)}_chunk_{i+1}", "txt")
                save_output_to_file(cleaned_chunk_summary, os.path.join(SUMMARIES_DIR, chunk_filename))
                chunk_summaries.append(cleaned_chunk_summary)

            except Exception as e:
                logging.error(f"Error processing chunk {i+1}/{len(chunks)} for file '{file_path}': {e}")
                copy_unreadable_file(file_path, 'repo', UNPROCESSED_DIR)
                return None, True

        final_summary = summarize_chunk_summaries(chunk_summaries, file_path, summarization_model, summarization_tokenizer_name)
        return final_summary, False
    else:
        summary = generate_response_with_ollama(prompt, summarization_model)
        return clean_generated_summary(summary), False

def is_test_file(file_path: str) -> bool:
    """Determine if the file is a test file."""
    file_path_lower = file_path.lower()
    test_indicators = [
        "src/it",
        "src/performance",
        "src/ct",
        "src/test",
        "test/resources",
        "test/java",
        "test",
        "/tests/",
        "\\tests\\",
        "/test/",
        "\\test\\",
        "test_",
        "_test",
        "spec/",
        "spec\\",
        "specs/",
        "specs\\",
        "/spec/",
        "\\spec\\",
    ]
    for indicator in test_indicators:
        if indicator in file_path_lower:
            return True

    return False

def is_relevant_file(file_path: str) -> bool:
    """Determine if a file is relevant to process."""
    # Exclude test files
    if is_test_file(file_path):
        return False
    # Exclude specific files
    EXCLUDED_FILES = [
        'pom.xml', 'jenkinsfile', 'build.gradle', 'package.json', 'package-lock.json',
        'yarn.lock', 'Makefile', 'Dockerfile', 'README.md', 'LICENSE', 'CONTRIBUTING.md',
        '.gitignore', 'gradlew', 'gradlew.bat', 'mvnw', 'mvnw.cmd', 'setup.py',
        'requirements.txt', 'environment.yml', 'Pipfile', 'Pipfile.lock', 'Gemfile', 'Gemfile.lock', '.gitlab-ci.yml', 'renovate.json'
    ]
    if os.path.basename(file_path).lower() in EXCLUDED_FILES:
        return False
    # Include files with relevant extensions
    RELEVANT_EXTENSIONS = [
        '.java', '.kt', '.xml', '.yml', '.yaml', '.properties', '.conf', '.sql', '.json',
        '.js', '.ts', '.tsx', '.jsx', '.py', '.rb', '.go', '.php', '.cs', '.cpp', '.c',
        '.h', '.swift', '.rs', '.erl', '.ex', '.exs', '.html', '.css'
    ]
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in RELEVANT_EXTENSIONS:
        return True
    return False

def copy_unreadable_file(file_path: str, base_directory: str, unprocessed_directory: str):
    """Copy unreadable files to a new directory while maintaining relative paths."""
    relative_path = os.path.relpath(file_path, base_directory)
    dest_path = os.path.join(unprocessed_directory, relative_path)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(file_path, dest_path)
    logging.info(f"Copied unreadable file {file_path} to {dest_path}")

def list_all_files(directory: str) -> list:
    """List all relevant files in the directory."""
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_relevant_file(file_path):
                all_files.append(file_path)
    return all_files

def read_file(file_path: str, base_directory: str, unprocessed_directory: str) -> str:
    """Read the contents of a file with robust encoding handling."""
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'

        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        copy_unreadable_file(file_path, base_directory, unprocessed_directory)
        return ""

def save_output_to_file(content: str, file_name: str):
    """Save output to a file."""
    with open(file_name, 'w') as f:
        f.write(content)

def read_hf_token(token_file: str) -> str:
    """Read HuggingFace token from a file."""
    try:
        with open(token_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"HuggingFace token file '{token_file}' not found.")
        exit(1)
    except Exception as e:
        logging.error(f"Error reading HuggingFace token file '{token_file}': {e}")
        exit(1)

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize codebase and generate PlantUML diagram.')
    parser.add_argument('--directory', type=str, default='repo', help='Directory of the codebase to summarize.')
    parser.add_argument('--summarization_model', type=str, default=DEFAULT_SUMMARIZATION_MODEL, help='Model to use for summarization.')
    parser.add_argument('--plantuml_model', type=str, default=DEFAULT_PLANTUML_MODEL, help='Model to use for generating the PlantUML diagram.')
    parser.add_argument('--summarization_tokenizer', type=str, default=DEFAULT_SUMMARIZATION_TOKENIZER_NAME, help='Tokenizer for summarization model.')
    parser.add_argument('--plantuml_tokenizer', type=str, default=DEFAULT_PLANTUML_TOKENIZER_NAME, help='Tokenizer for PlantUML model.')
    parser.add_argument('--hf_token_file', type=str, default='hf_token.txt', help='Path to the HuggingFace token file.')
    parser.add_argument('--plantuml_context_file', type=str, default=None, help='Path to the file containing the PlantUML context (prompt).')
    args = parser.parse_args()

    # Read HuggingFace token from file
    hf_token = read_hf_token(args.hf_token_file)
    if hf_token:
        login(token=hf_token)
    else:
        logging.error("HuggingFace API token is not set.")
        exit(1)

    # Read PlantUML context from file or use default
    if args.plantuml_context_file:
        try:
            with open(args.plantuml_context_file, 'r') as f:
                plantuml_context = f.read()
        except Exception as e:
            logging.error(f"Error reading PlantUML context file '{args.plantuml_context_file}': {e}")
            exit(1)
    else:
        plantuml_context = DEFAULT_PLANTUML_PROMPT_TEMPLATE

    directory = args.directory
    summarization_model = args.summarization_model
    plantuml_model = args.plantuml_model
    summarization_tokenizer_name = args.summarization_tokenizer
    plantuml_tokenizer_name = args.plantuml_tokenizer

    codebase_summary = summarize_codebase(
        directory,
        summarization_model,
        summarization_tokenizer_name,
        plantuml_model,
        plantuml_tokenizer_name,
        plantuml_context
    )

    if codebase_summary:
        logging.info("Final codebase summary generated.")
    else:
        logging.warning("No files found or summarized.")
