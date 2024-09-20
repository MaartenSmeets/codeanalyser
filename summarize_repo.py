import os
import shelve
import uuid
import argparse
import logging
import traceback
import chardet
import shutil
import subprocess
from hashlib import md5
from transformers import AutoTokenizer
from ollama import generate
from datetime import datetime
from huggingface_hub import login

# Configuration: Default models and context length
DEFAULT_SUMMARIZATION_MODEL = 'gemma2:9b-instruct-q8_0'
DEFAULT_PLANTUML_MODEL = 'deepseek-coder-v2:16b-lite-base-fp16'
DEFAULT_TOKENIZER_NAME = "google/gemma-2-9b-it"
DEFAULT_PLANTUML_TOKENIZER_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
CACHE_DIR = 'llm_cache'
MAX_CONTEXT_LENGTH = 8192  # Maximum token length for a single prompt
SUMMARIES_DIR = "summaries"
UNPROCESSED_DIR = "unprocessed_files"
PLANTUML_DIR = "plantuml"
PLANTUML_FILE = "codebase_diagram.puml"
PLANTUML_PNG_FILE = "codebase_diagram.png"

# Authenticate to HuggingFace using the token
hf_token = "?"
if hf_token:
    login(token=hf_token)
else:
    logging.error("HuggingFace API token is not set.")
    exit(1)

# Configure logging with timestamps, writing to a log file that is overwritten on each new run
log_file = 'script_run.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])

# Function to initialize the shelve cache
def init_cache():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return shelve.open(os.path.join(CACHE_DIR, 'llm_cache.db'))

# Function to clean and format the final summary
def clean_generated_summary(summary):
    # Remove sentences that start with "Let me know"
    cleaned_summary = "\n".join(
        [sentence for sentence in summary.split("\n") if not sentence.startswith("Let me know")]
    )
    # Remove trailing empty newlines
    return cleaned_summary.rstrip()

# Generate a unique hash for cache key based on input
def generate_cache_key(prompt, model):
    key_string = f"{model}_{prompt}"
    return md5(key_string.encode()).hexdigest()

# Function to initialize tokenizer based on the model
def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True)
    return tokenizer

# Function to split text into chunks based on token length
def split_into_chunks(text, max_tokens, tokenizer):
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

    logging.info(f"File split into {len(chunks)} chunks.")  # Log the number of chunks created
    return chunks

# Helper function to generate unique filenames
def generate_unique_filename(base_name, extension):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# Function to call the LLM via Ollama to generate summaries (with caching via shelve)
def generate_response_with_ollama(prompt, model, tokenizer_name):
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
        response_content = generate(
            model=model, prompt=prompt).get('response', '')

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
        raise e  # Re-raise the exception to allow the calling function to handle it.

# Function to summarize chunks and save the result
def summarize_chunk_summaries(chunk_summaries, file_path, summarization_model, summarization_tokenizer_name):
    chunk_summary_text = "\n\n".join(chunk_summaries)

    prompt = f"""
You have summarized a large file in multiple chunks. Now, based on the following individual chunk summaries, create a single cohesive, structured, and factual summary of the entire file. Strictly and only provide the summary. Do not summarize the summary. Do not ask for confirmation. Do not provide suggestions. Do not suggest or ask additional input or follow-up questions. Do not indicate potential uses or interpretations. Do not provide recommendations.

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
{chunk_summary_text}
"""

    final_summary = generate_response_with_ollama(prompt, summarization_model, summarization_tokenizer_name)
    return clean_generated_summary(final_summary)

# Function to extract PlantUML code from LLM output
def extract_plantuml_code(content):
    start_tag = "@startuml"
    end_tag = "@enduml"
    start_index = content.find(start_tag)
    end_index = content.find(end_tag)
    if start_index != -1 and end_index != -1:
        # Include the end_tag in the output
        end_index += len(end_tag)
        return content[start_index:end_index]
    else:
        logging.warning("PlantUML tags @startuml and @enduml not found in the LLM output.")
        return content  # Return content as is if tags not found

# Function to summarize the entire repository and generate the PlantUML diagram
def summarize_codebase(directory, summarization_model, summarization_tokenizer_name, plantuml_model, plantuml_tokenizer_name):
    all_files = list_all_files(directory)
    total_files = len(all_files)
    codebase_summary = []
    logging.info(f"Total files to process: {total_files}")

    if not os.path.exists(SUMMARIES_DIR):
        os.makedirs(SUMMARIES_DIR)

    if not os.path.exists(UNPROCESSED_DIR):
        os.makedirs(UNPROCESSED_DIR)

    # Enhanced error handling to provide better debug information
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
                # Include relative path and filename in the summary for reference
                cleaned_summary = clean_generated_summary(summary)
                formatted_summary = f"File: {file_path}\n\n{cleaned_summary}\n"
                codebase_summary.append(formatted_summary)
                file_summary_path = generate_unique_filename(
                    os.path.basename(file_path), "summary.txt")
                save_output_to_file(formatted_summary, os.path.join(
                    SUMMARIES_DIR, file_summary_path))

            if idx % 5 == 0 or idx == total_files:
                logging.info(f"Progress: {idx}/{total_files} files processed.")
        
        except Exception as e:
            # Enhanced error logging for more useful debug information
            logging.error(f"Error processing file: {file_path}")
            logging.error(f"Exception details: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

            # Copy the file to unprocessed_files if it fails
            copy_unreadable_file(file_path, directory, UNPROCESSED_DIR)

    # Combine all file summaries for the PlantUML diagram
    combined_summary = "\n".join(codebase_summary)
    
    if combined_summary:
        # Save the final codebase summary
        logging.info("Final codebase summary generated.")
        summary_file = generate_unique_filename("codebase_summary", "txt")
        save_output_to_file(combined_summary, summary_file)

        # Ask the LLM to generate the PlantUML diagram from the summary
        llm_plantuml_prompt = f"""Based on the following comprehensive codebase summary, generate a valid PlantUML diagram that accurately represents the system's architecture and functional flow. The diagram should be high-level, focusing on the major components, their interactions, and data flow.

Instructions:

- The output should be valid PlantUML code, enclosed within @startuml and @enduml.
- Use appropriate PlantUML elements such as packages, classes, interfaces, and arrows to represent the components and their relationships.
- Focus on the architectural and functional relationships between components.
- Ensure that the syntax is correct and that the diagram can be rendered without errors.
- Do not include any explanations or descriptions outside of the PlantUML code.

**Codebase Summary**:
{combined_summary}
"""

        # Call LLM to generate a PlantUML diagram structure
        plantuml_diagram_content = generate_response_with_ollama(llm_plantuml_prompt, plantuml_model, plantuml_tokenizer_name)

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

# Function to generate a summary for each file
def generate_summary(file_path, file_content, summarization_model, summarization_tokenizer_name):
    _, file_extension = os.path.splitext(file_path)

    if is_test_file(file_path):
        logging.info(f"Skipping test file: {file_path}")
        return None, True

    if len(file_content.strip()) == 0:
        logging.warning(f"Skipping empty file: {file_path}")
        return None, True

    prompt_template = f"""You are tasked with summarizing a file from a software repository. Provide a **precise**, **comprehensive**, and **well-structured** English summary that accurately reflects the contents of the file. Do not ask for confirmation. Do not provide suggestions or recommendations. Focus solely on creating the summary and keep it concise. Avoid redundancy and do not summarize the summary. The summary must be:

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
"""

    tokenizer = get_tokenizer(summarization_tokenizer_name)
    
    # Tokenize just the prompt template
    prompt_token_count = len(
        tokenizer(prompt_template, return_tensors="pt").input_ids[0]
    )

    logging.debug(f"Prompt token count for file '{file_path}': {prompt_token_count}")

    if prompt_token_count >= MAX_CONTEXT_LENGTH:
        logging.error(f"Prompt is too long for file '{file_path}' (token count: {prompt_token_count}). Skipping file.")
        return None, True

    # Now include the file content in the prompt
    prompt = f"{prompt_template}\n{file_content}"

    full_prompt_token_count = len(
        tokenizer(prompt, return_tensors="pt").input_ids[0]
    )

    logging.debug(f"Full prompt token count for file '{file_path}': {full_prompt_token_count}")

    if full_prompt_token_count > MAX_CONTEXT_LENGTH:
        logging.debug(f"File '{file_path}' exceeds context length; processing in chunks.")
        available_tokens_for_content = MAX_CONTEXT_LENGTH - prompt_token_count

        if available_tokens_for_content <= 0:
            logging.error(f"Not enough space for content in the context for file '{file_path}'. Skipping file.")
            return None, True

        chunks = split_into_chunks(file_content, available_tokens_for_content, tokenizer)
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            chunk_prompt = f"""
You are summarizing a portion of a file in a software repository. This portion belongs to a larger file, and it is part {i+1} of {len(chunks)}. Summarize strictly and solely the content of this chunk in a clear, structured, and concise manner, focusing on relevant technical and functional details. Do not provide recommendations or ask follow-up questions.

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
{chunk}
"""

            logging.info(f"Processing chunk {i+1}/{len(chunks)} for file '{file_path}'")
            
            try:
                chunk_summary = generate_response_with_ollama(chunk_prompt, summarization_model, summarization_tokenizer_name)
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
        summary = generate_response_with_ollama(prompt, summarization_model, summarization_tokenizer_name)
        return clean_generated_summary(summary), False

# Function to determine if the file is a test file
def is_test_file(file_path):
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

# Function to determine if a file is relevant to process
def is_relevant_file(file_path):
    # Exclude test files
    if is_test_file(file_path):
        return False
    # Exclude specific files
    EXCLUDED_FILES = [
        'pom.xml', 'jenkinsfile', 'build.gradle', 'package.json', 'package-lock.json',
        'yarn.lock', 'Makefile', 'Dockerfile', 'README.md', 'LICENSE', 'CONTRIBUTING.md',
        '.gitignore', 'gradlew', 'gradlew.bat', 'mvnw', 'mvnw.cmd', 'setup.py',
        'requirements.txt', 'environment.yml', 'Pipfile', 'Pipfile.lock', 'Gemfile', 'Gemfile.lock', '.gitlab-ci.yml'
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

# Function to copy unreadable files to a new directory while maintaining relative paths
def copy_unreadable_file(file_path, base_directory, unprocessed_directory):
    relative_path = os.path.relpath(file_path, base_directory)
    dest_path = os.path.join(unprocessed_directory, relative_path)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(file_path, dest_path)
    logging.info(f"Copied unreadable file {file_path} to {dest_path}")

# Function to list all files in the directory
def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_relevant_file(file_path):
                all_files.append(file_path)
    return all_files

# Function to read the contents of a file with robust encoding handling
def read_file(file_path, base_directory, unprocessed_directory):
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

# Function to save output to a file
def save_output_to_file(content, file_name):
    with open(file_name, 'w') as f:
        f.write(content)

# Main entry point for summarizing the codebase
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize codebase and generate PlantUML diagram.')
    parser.add_argument('--directory', type=str, default='repo', help='Directory of the codebase to summarize.')
    parser.add_argument('--summarization_model', type=str, default=DEFAULT_SUMMARIZATION_MODEL, help='Model to use for summarization.')
    parser.add_argument('--plantuml_model', type=str, default=DEFAULT_PLANTUML_MODEL, help='Model to use for generating the PlantUML diagram.')
    parser.add_argument('--summarization_tokenizer', type=str, default=DEFAULT_TOKENIZER_NAME, help='Tokenizer for summarization model.')
    parser.add_argument('--plantuml_tokenizer', type=str, default=DEFAULT_PLANTUML_TOKENIZER_NAME, help='Tokenizer for PlantUML model.')
    args = parser.parse_args()

    directory = args.directory
    summarization_model = args.summarization_model
    plantuml_model = args.plantuml_model
    summarization_tokenizer_name = args.summarization_tokenizer
    plantuml_tokenizer_name = args.plantuml_tokenizer

    codebase_summary = summarize_codebase(directory, summarization_model, summarization_tokenizer_name, plantuml_model, plantuml_tokenizer_name)

    if codebase_summary:
        logging.info("Final codebase summary generated.")
    else:
        logging.warning("No files found or summarized.")
