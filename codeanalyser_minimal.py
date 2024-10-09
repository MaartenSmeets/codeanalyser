import os
import shelve
import uuid
import logging
import traceback
import chardet
import shutil
import json
import requests
import re
from hashlib import md5
from transformers import AutoTokenizer
from datetime import datetime
from huggingface_hub import login

# Constants and Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"  # Configurable Ollama URL

DEFAULT_SUMMARIZATION_MODEL = 'gemma2:9b-instruct-q8_0'
DEFAULT_SUMMARIZATION_TOKENIZER_NAME = 'google/gemma-2-9b-it'
MAX_SUMMARIZATION_CONTEXT_LENGTH = 7000

CACHE_DIR = 'llm_cache'
SUMMARIES_DIR = "summaries"
IRRELEVANT_SUMMARIES_DIR = "irrelevant_summaries"
UNPROCESSED_DIR = "unprocessed_files"
MERMAID_PROMPT_FILE = "mermaid_prompt.txt"

# Logging Configuration
log_file = 'script_run.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
)

# Prompt Templates
DEFAULT_MERMAID_PROMPT_TEMPLATE = """**Objective:**
Based on the provided detailed codebase summary, generate a **Mermaid diagram** that clearly represents the system's 
architecture, major components, and data flow in a visually appealing and easy-to-understand manner. Focus on illustrating 
the **logical grouping of components**, their **interactions**, and the **data flow** between both internal and external 
systems. Make sure not to use special characters. You are only allowed in names, groupings, edges, nodes, etc to use 
alphanumeric characters. Also avoid mentioning file extensions and function parameters. Avoid mentioning filenames directly 
and use a functional name instead.

**Instructions:**

- **Generate valid Mermaid code** that accurately reflects the system architecture.
- Focus on **major components** and their **functional groupings**. Avoid mentioning individual files and solely technical components (unless they are external dependency). Do not be overly detailed.
- Use **clear, descriptive labels** for both nodes and edges to make the diagram intuitive for stakeholders.
- **Organize components into subgraphs** or groups based on logical relationships (e.g., services, databases, external APIs) 
  and use distinct colors in the diagram for logical groups to provide a clear and structured view.
- Use a flowchart with left to right layout for enhanced readability.
- Maintain **consistent visual patterns** to distinguish between types of components.
- **Apply a minimal color scheme** to differentiate between logical groupings, system layers or types of components, keeping the design professional.
- Use **edge labels** to describe the nature of interactions or data flow between components (e.g., "sends data", "receives response", "queries database").
- **Minimize crossing edges** and ensure proper spacing to avoid clutter and maintain clarity.
- Ensure the Mermaid syntax is correct, and the diagram can be rendered without errors.

---

**Input:**  
- A comprehensive codebase summary in the form: `{combined_summary}`

**Your Task:**  
Generate a **well-structured and visually appealing** Mermaid diagram that illustrates the system’s architecture and functional data flows based on the provided summary. The output should be valid Mermaid code, with no extra commentary or text beyond the code itself.
"""

FILE_SUMMARY_PROMPT_TEMPLATE = """You are tasked with summarizing a file from a software repository. Provide only a **precise**, **comprehensive**, and **well-structured** English summary that accurately reflects the contents of the file. Do not write or update code. Do not generate code to create a summary but create a summary. Do not ask for confirmation. Do not provide suggestions. Do not provide recommendations. Do not mention potential improvements. Do not mention considerations. Focus solely on creating the summary. Avoid redundancy and do not summarize the summary. The summary must be:

- **Factual and objective**: Include only verifiable information based on the provided file. Avoid any assumptions, opinions, interpretations, or speculative conclusions.
- **Specific and relevant**: Directly reference the actual contents of the file. Avoid general statements or unrelated information. Focus on the specific purpose, functionality, and structure of the file.
- **Concise yet complete**: Ensure that the summary captures all essential details while being succinct. Eliminate redundancy and unnecessary information.

In particular, address the following when applicable and relevant to the file’s role in the codebase. When not applicable, leave out the section:
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
- Exclude any assumptions, opinions, or evaluations and do not provide concluding remarks.

**Filename**: 
{file_path}

**Chunk content**:
{chunk_content}

***Your task**
The goal is to accurately capture the functionality and purpose of this specific chunk within the file.
"""

EVALUATE_RELEVANCE_PROMPT = """
Based on the following summary, determine whether the file is relevant for generating an architectural diagram of the codebase. When in doubt, consider the file **relevant**. Include any file that could remotely contribute to understanding the architecture, even if its relevance is not immediately obvious.

A file should be considered relevant if it:

- Is **central to the system's architecture** or defines key components.
- Provides **functional descriptions** of services, components, modules, or how the system operates and interacts with other parts of the architecture.
- Describes **data flows**, such as the movement, transformation, or interaction of data between components or systems.
- Includes **jobs, tasks, or processes** that are important to the system’s operations, even if they are not directly tied to core architectural components.

When unsure, or if there is any possibility the file could aid in representing the system architecture, respond with 'Yes'. 

Summary:
{summary}

Is this file relevant for generating the architectural diagram? Respond with 'Yes' or 'No' only.
"""

# Helper Functions
def replace_special_statements(text):
    pattern = r'(\|\s*([^\|]+)\s*\|)|(\[\s*([^\]]+)\s*\])|(\{\s*([^\}]+)\s*\})'
    
    def replace_func(match):
        if match.group(1):  # If the match is within | |
            content = match.group(2)
        elif match.group(3):  # If the match is within [ ]
            content = match.group(4)
        else:  # If the match is within { }
            content = match.group(6)
        
        cleaned_content = ''.join(word.capitalize() for word in re.findall(r'\w+', content))
        
        if match.group(1):  # Replace | |
            return f"|{cleaned_content}|"
        elif match.group(3):  # Replace [ ]
            return f"[{cleaned_content}]"
        else:  # Replace { }
            return f"{{{cleaned_content}}}"
    
    return re.sub(pattern, replace_func, text)

def generate_unique_filename(base_name: str, extension: str) -> str:
    """Generate a unique filename with timestamp and unique ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    safe_base_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', base_name)  # Make the base name safe for filenames
    return f"{safe_base_name}_{timestamp}_{unique_id}.{extension}"

def init_cache() -> shelve.Shelf:
    """Initialize the shelve cache."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return shelve.open(os.path.join(CACHE_DIR, 'llm_cache.db'))

def generate_cache_key(prompt: str, model: str) -> str:
    """Generate a unique hash for cache key based on input."""
    key_string = f"{model}_{prompt}"
    return md5(key_string.encode()).hexdigest()

def get_tokenizer(tokenizer_name: str):
    """Initialize tokenizer based on the model."""
    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

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

def generate_response_with_ollama(prompt: str, model: str) -> str:
    """Call the LLM via Ollama to generate responses with caching."""
    cache = init_cache()
    cache_key = generate_cache_key(prompt, model)

    # Check if the result is already cached
    if cache_key in cache:
        logging.info(f"Fetching result from cache for prompt: {prompt[:50]}...")
        response_content = cache[cache_key]
        cache.close()
        return response_content

    try:
        logging.debug(f"Sending request to Ollama with model '{model}' and prompt size {len(prompt)}")
        url = OLLAMA_URL
        payload = {"model": model, "prompt": prompt}
        headers = {'Content-Type': 'application/json'}

        # Send the request
        response = requests.post(url, data=json.dumps(payload), headers=headers, stream=True)

        if response.status_code != 200:
            logging.error(f"Failed to generate response with Ollama: HTTP {response.status_code}")
            return ""

        response_content = ''
        for line in response.iter_lines():
            if line:
                try:
                    line_json = json.loads(line.decode('utf-8'))
                    response_text = line_json.get('response', '')
                    response_content += response_text
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse line as JSON: {e}")
                    continue

        if not response_content:
            logging.warning("Unexpected response or no response.")
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

def extract_mermaid_code(content: str) -> str:
    """Extract Mermaid code from LLM output."""
    lines = content.strip().splitlines()

    # Remove leading and trailing ``` if present
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()

def generate_mermaid_code(combined_summary: str, mermaid_context: str) -> str:
    """Generate the Mermaid code based on the combined summary."""
    llm_mermaid_prompt = mermaid_context.format(combined_summary=combined_summary)

    # Save the Mermaid prompt context to a file for debugging
    save_output_to_file(llm_mermaid_prompt, MERMAID_PROMPT_FILE)
    return

def summarize_codebase(directory: str, summarization_model: str, summarization_tokenizer_name: str) -> str:
    """Summarize the entire repository."""
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

    for idx, file_path in enumerate(all_files, start=1):
        logging.info(f"Processing file {idx}/{total_files}: {file_path}")
        file_content = read_file(file_path, directory, UNPROCESSED_DIR)

        if not file_content:
            logging.warning(f"Skipping unreadable or empty file: {file_path}")
            continue

        try:
            summary, is_test_file_flag, was_chunked = generate_summary(file_path, file_content, summarization_model, summarization_tokenizer_name)

            if is_test_file_flag:
                logging.info(f"Test or irrelevant file detected and skipped: {file_path}")
                continue

            if summary:
                is_relevant = evaluate_relevance(summary, summarization_model)

                cleaned_summary = clean_generated_summary(summary)
                if was_chunked:
                    formatted_summary = f"File: {file_path}\n(The following summaries were concatenated from multiple chunks)\n\n{cleaned_summary}\n"
                else:
                    formatted_summary = f"File: {file_path}\n\n{cleaned_summary}\n"

                if is_relevant:
                    codebase_summary.append(formatted_summary)
                    file_summary_path = generate_unique_filename(os.path.basename(file_path), "summary.txt")
                    save_output_to_file(formatted_summary, os.path.join(SUMMARIES_DIR, file_summary_path))
                else:
                    logging.info(f"File '{file_path}' deemed not relevant for the diagram.")
                    file_summary_path = generate_unique_filename(os.path.basename(file_path), "summary.txt")
                    save_output_to_file(formatted_summary, os.path.join(IRRELEVANT_SUMMARIES_DIR, file_summary_path))

            if idx % 5 == 0 or idx == total_files:
                logging.info(f"Progress: {idx}/{total_files} files processed.")

        except Exception as e:
            logging.error(f"Error processing file: {file_path}")
            logging.error(f"Exception details: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

            copy_unreadable_file(file_path, directory, UNPROCESSED_DIR)

    combined_summary = "\n".join(codebase_summary)

    if combined_summary:
        logging.info("Final codebase summary generated.")
        summary_file = generate_unique_filename("codebase_summary", "txt")
        save_output_to_file(combined_summary, summary_file)
    else:
        logging.warning("No relevant summaries generated.")

    return combined_summary

def generate_summary(file_path: str, file_content: str, summarization_model: str, summarization_tokenizer_name: str) -> tuple:
    """Generate a summary for each file."""
    if is_test_file(file_path):
        logging.info(f"Skipping test file: {file_path}")
        return None, True, False

    if len(file_content.strip()) == 0:
        logging.warning(f"Skipping empty file: {file_path}")
        return None, True, False

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
            return None, True, False

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
                return None, True, False

        final_summary = "\n".join(chunk_summaries)
        return final_summary, False, True
    else:
        summary = generate_response_with_ollama(prompt, summarization_model)
        return clean_generated_summary(summary), False, False

def clean_generated_summary(summary: str) -> str:
    """Clean and format the final summary."""
    cleaned_summary = "\n".join(
        [sentence for sentence in summary.split("\n") if not sentence.startswith("Let me know")]
    )
    return cleaned_summary.rstrip()

def evaluate_relevance(summary: str, summarization_model: str) -> bool:
    """Evaluate if the summary is relevant for generating the diagram."""
    prompt = EVALUATE_RELEVANCE_PROMPT.format(summary=summary)
    response = generate_response_with_ollama(prompt, summarization_model)
    response_cleaned = response.strip().lower()
    return response_cleaned.startswith('yes')

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

def list_all_files(directory: str) -> list:
    """List all relevant files in the directory."""
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_relevant_file(file_path):
                all_files.append(file_path)
    return all_files

def copy_unreadable_file(file_path: str, base_directory: str, unprocessed_directory: str):
    """Copy unreadable files to a new directory while maintaining relative paths."""
    relative_path = os.path.relpath(file_path, base_directory)
    dest_path = os.path.join(unprocessed_directory, relative_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(file_path, dest_path)
    logging.info(f"Copied unreadable file {file_path} to {dest_path}")

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
        'requirements.txt', 'environment.yml', 'Pipfile', 'Pipfile.lock', 'Gemfile', 
        'Gemfile.lock', '.gitlab-ci.yml', 'renovate.json', 'Dockerfile','docker-compose.yml',
        'bootstrap.min.css'
    ]
    if os.path.basename(file_path).lower() in EXCLUDED_FILES:
        return False
    # Include files with relevant extensions
    RELEVANT_EXTENSIONS = [
        '.java', '.kt', '.xml', '.yml', '.yaml', '.properties', '.conf', '.sql', '.json',
        '.js', '.ts', '.tsx', '.jsx', '.py', '.rb', '.go', '.php', '.cs', '.cpp', '.c',
        '.h', '.swift', '.rs', '.erl', '.ex', '.exs', '.html'
    ]
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in RELEVANT_EXTENSIONS:
        return True
    return False

def main():
    # Define constants for the directory and models
    directory = 'repo'
    summarization_model = DEFAULT_SUMMARIZATION_MODEL
    summarization_tokenizer_name = DEFAULT_SUMMARIZATION_TOKENIZER_NAME
    mermaid_context = DEFAULT_MERMAID_PROMPT_TEMPLATE

    # Run summarization
    codebase_summary = summarize_codebase(
        directory,
        summarization_model,
        summarization_tokenizer_name,
    )

    if codebase_summary:
        # Generate Mermaid code
        generate_mermaid_code(
            codebase_summary,
            mermaid_context
        )
    else:
        logging.warning("No files found or summarized.")


if __name__ == "__main__":
    main()
