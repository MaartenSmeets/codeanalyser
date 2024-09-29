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

DEFAULT_SUMMARIZATION_MODEL = 'gemma2:9b-instruct-q8_0'#'mixtral:8x7b-instruct-v0.1-q8_0'
DEFAULT_SUMMARIZATION_TOKENIZER_NAME = 'google/gemma-2-9b-it'#"mistralai/Mixtral-8x7B-Instruct-v0.1"
MAX_SUMMARIZATION_CONTEXT_LENGTH = 7000  # Max token length for summarization model

DEFAULT_MERMAID_MODEL = 'mixtral:8x7b-instruct-v0.1-q4_K_M'
DEFAULT_MERMAID_TOKENIZER_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
MAX_MERMAID_CONTEXT_LENGTH = 24000  # Max token length for Mermaid model

CACHE_DIR = 'llm_cache'
SUMMARIES_DIR = "summaries"
IRRELEVANT_SUMMARIES_DIR = "irrelevant_summaries"
UNPROCESSED_DIR = "unprocessed_files"
MERMAID_FILE = "codebase_diagram.mmd"
MERMAID_PNG_FILE = "codebase_diagram.png"
MERMAID_PROMPT_FILE = "mermaid_prompt.txt"
MERMAID_FIX_PROMPT_DIR = "mermaid_fix_prompts"
MAX_FIX_RETRIES = 3

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
- Focus on **major components** and their **functional groupings**, avoiding individual files or overly detailed elements.
- Use **clear, descriptive labels** for both nodes and edges to make the diagram intuitive for stakeholders.
- **Organize components into subgraphs** or groups based on logical relationships (e.g., services, databases, external APIs) 
  to provide a clear and structured view.
- Use appropriate **Mermaid diagram types** (e.g., flowcharts, sequence diagrams) that best represent the architecture.
- Maintain **consistent visual patterns** to distinguish between types of components.
- Arrange the diagram to **flow from left to right** or **top to bottom** for enhanced readability.
- **Apply a minimal color scheme** to differentiate between system layers or types of components, keeping the design professional.
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

Remember you are tasked with creating a detailed summary of the file content only.
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

MERMAID_FIX_PROMPT_TEMPLATE = """You are provided with a Mermaid diagram code that contains syntax errors causing it to fail rendering into a PNG image. Your task is to fix any Mermaid syntax errors without altering the content or meaning of the diagram. Do not change the overall structure or components represented. Often a case of errors is the usage of special characters in names such as spaces dots and quotes. Fix those. For example replace ("This is Text") with ThisIsText and thus removing these special characters.

**Instructions:**

- Carefully examine the provided Mermaid code and the error message.
- Identify and correct any syntax errors in the Mermaid code.
- Ensure the corrected Mermaid code can be successfully rendered into a PNG image without errors.
- Provide only the corrected Mermaid code without any additional explanation or comments.
- Do not enclose the Mermaid code in triple backticks (```) or any other formatting.
---

**Mermaid Code with Errors:**
{mermaid_code}

**Error Message:**
{error_message}

**Your Task:**
Provide only the corrected Mermaid code below. Do not include any additional text, comments or final remarks.
"""

# Helper Functions
def replace_special_statements(text):
    # Regex pattern for | |, [ ], and { }
    pattern = r'(\|\s*([^\|]+)\s*\|)|(\[\s*([^\]]+)\s*\])|(\{\s*([^\}]+)\s*\})'
    
    # Function to clean up the text within the delimiters and make it alphanumeric
    def replace_func(match):
        if match.group(1):  # If the match is within | |
            content = match.group(2)
        elif match.group(3):  # If the match is within [ ]
            content = match.group(4)
        else:  # If the match is within { }
            content = match.group(6)
        
        # Remove all non-alphanumeric characters (including dots) and capitalize each word
        cleaned_content = ''.join(word.capitalize() for word in re.findall(r'\w+', content))
        
        # Return the new formatted string based on the matched pattern
        if match.group(1):  # Replace | |
            return f"|{cleaned_content}|"
        elif match.group(3):  # Replace [ ]
            return f"[{cleaned_content}]"
        else:  # Replace { }
            return f"{{{cleaned_content}}}"
    
    # Substitute all matches in the text
    return re.sub(pattern, replace_func, text)


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
    safe_base_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', base_name)
    return f"{safe_base_name}_{timestamp}_{unique_id}.{extension}"

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

def extract_mermaid_code(content: str) -> str:
    """
    Extract Mermaid code from LLM output.
    If ``` is detected in the last several lines, remove it and all lines after.
    """
    lines = content.strip().splitlines()

    # Remove any ``` and lines after it in the last several lines
    for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):
        if lines[i].strip().startswith("```"):
            lines = lines[:i]
            break

    # Remove leading and trailing ``` if present
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()

def evaluate_relevance(summary: str, summarization_model: str) -> bool:
    """Evaluate if the summary is relevant for generating the diagram."""
    prompt = EVALUATE_RELEVANCE_PROMPT.format(summary=summary)

    response = generate_response_with_ollama(prompt, summarization_model)
    response_cleaned = response.strip().lower()
    return response_cleaned.startswith('yes')

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

    # Process each file
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
                # Evaluate relevance
                is_relevant = evaluate_relevance(summary, summarization_model)

                # Include relative path and filename in the summary for reference
                cleaned_summary = clean_generated_summary(summary)
                if was_chunked:
                    formatted_summary = f"File: {file_path}\n(The following summaries were concatenated from multiple chunks)\n\n{cleaned_summary}\n"
                else:
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

    # Combine all relevant file summaries
    combined_summary = "\n".join(codebase_summary)

    if combined_summary:
        # Save the final codebase summary
        logging.info("Final codebase summary generated.")
        summary_file = generate_unique_filename("codebase_summary", "txt")
        save_output_to_file(combined_summary, summary_file)
    else:
        logging.warning("No relevant summaries generated.")

    return combined_summary

def generate_mermaid_code(combined_summary: str, mermaid_context: str, mermaid_model: str, mermaid_tokenizer_name: str) -> str:
    """Generate the Mermaid code based on the combined summary."""
    # Generate the Mermaid prompt
    llm_mermaid_prompt = mermaid_context.format(combined_summary=combined_summary)

    # Save the Mermaid prompt context to a file for debugging
    save_output_to_file(llm_mermaid_prompt, MERMAID_PROMPT_FILE)

    logging.info("Preparing to generate Mermaid diagram...")

    # Check if the prompt exceeds the maximum context length
    mermaid_tokenizer = get_tokenizer(mermaid_tokenizer_name)
    prompt_token_count = len(
        mermaid_tokenizer(llm_mermaid_prompt, return_tensors="pt").input_ids[0]
    )

    if prompt_token_count > MAX_MERMAID_CONTEXT_LENGTH:
        logging.error("Mermaid prompt exceeds maximum context length. Cannot generate diagram.")
        raise ValueError("Mermaid prompt exceeds maximum context length.")

    logging.info("Restarting Ollama to free up memory before generating Mermaid diagram...")

    # Restart Ollama to free up memory before generating Mermaid diagram
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

    logging.info("Generating Mermaid diagram...")

    # Call LLM to generate a Mermaid diagram structure
    mermaid_diagram_content = generate_response_with_ollama(llm_mermaid_prompt, mermaid_model)

    # Extract valid Mermaid code from the LLM output
    mermaid_code = extract_mermaid_code(mermaid_diagram_content)
    mermaid_code = replace_special_statements(mermaid_code)

    return mermaid_code

def process_mermaid_diagram(mermaid_code: str, mermaid_model: str):
    """Process the Mermaid code and attempt to generate a PNG, fixing errors if necessary."""
    # Pre-process existing Mermaid file if it exists
    if os.path.exists(MERMAID_FILE):
        logging.info(f"{MERMAID_FILE} exists. Pre-processing with replace_special_statements.")
        # Read existing Mermaid code from the file
        with open(MERMAID_FILE, 'r') as f:
            existing_mermaid_code = f.read()
        
        # Apply the replace_special_statements function to process the code
        existing_mermaid_code = replace_special_statements(existing_mermaid_code)
        
        # Proceed with further processing on the updated Mermaid code
        mermaid_code = existing_mermaid_code
    
    # Initialize retry counter
    retry_count = 0
    success = False

    while retry_count < MAX_FIX_RETRIES and not success:
        # Save the Mermaid diagram content to a file
        with open(MERMAID_FILE, 'w') as f:
            f.write(mermaid_code)

        # Try to convert the Mermaid diagram to PNG using Mermaid CLI
        try:
            result = subprocess.run(
                ["mmdc", "-i", MERMAID_FILE, "-o", MERMAID_PNG_FILE, "-s", "5"],
                check=True,
                capture_output=True,
                text=True,
                env={"PUPPETEER_EXECUTABLE_PATH": "/usr/bin/chromium"}
            )
            logging.info(f"Mermaid diagram saved as PNG.")
            success = True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate Mermaid PNG: {e}")
            error_message = e.stderr  # Capture the standard error output
            logging.debug(f"Standard error from mmdc:\n{error_message}")
            # Prepare prompt to fix the diagram
            fix_prompt = MERMAID_FIX_PROMPT_TEMPLATE.format(
                mermaid_code=mermaid_code,
                error_message=error_message
            )
            # Save the fix prompt to a unique file
            fix_prompt_file = generate_unique_filename("mermaid_fix_prompt", "txt")
            fix_prompt_path = os.path.join(MERMAID_FIX_PROMPT_DIR, fix_prompt_file)
            save_output_to_file(fix_prompt, fix_prompt_path)
            logging.info(f"Saved Mermaid fix prompt to {fix_prompt_path}")

            # Call LLM to fix the Mermaid diagram
            logging.info(f"Asking LLM to fix the Mermaid diagram (Attempt {retry_count + 1})...")
            mermaid_code_fixed = generate_response_with_ollama(fix_prompt, mermaid_model)
            # Update the mermaid_code with the fixed version
            mermaid_code = extract_mermaid_code(mermaid_code_fixed)
            retry_count += 1

    if not success:
        logging.error("Failed to generate Mermaid diagram after multiple attempts.")
    else:
        logging.info("Successfully generated Mermaid diagram after fixing errors.")


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
        'requirements.txt', 'environment.yml', 'Pipfile', 'Pipfile.lock', 'Gemfile', 'Gemfile.lock', '.gitlab-ci.yml', 'renovate.json', 'Dockerfile','docker-compose.yml'
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
    parser = argparse.ArgumentParser(description='Summarize codebase and generate Mermaid diagram.')
    parser.add_argument('--directory', type=str, default='repo', help='Directory of the codebase to summarize.')
    parser.add_argument('--summarization_model', type=str, default=DEFAULT_SUMMARIZATION_MODEL, help='Model to use for summarization.')
    parser.add_argument('--mermaid_model', type=str, default=DEFAULT_MERMAID_MODEL, help='Model to use for generating the Mermaid diagram.')
    parser.add_argument('--summarization_tokenizer', type=str, default=DEFAULT_SUMMARIZATION_TOKENIZER_NAME, help='Tokenizer for summarization model.')
    parser.add_argument('--mermaid_tokenizer', type=str, default=DEFAULT_MERMAID_TOKENIZER_NAME, help='Tokenizer for Mermaid model.')
    parser.add_argument('--hf_token_file', type=str, default='hf_token.txt', help='Path to the HuggingFace token file.')
    parser.add_argument('--mermaid_context_file', type=str, default=None, help='Path to the file containing the Mermaid context (prompt).')
    args = parser.parse_args()

    # Read HuggingFace token from file
    hf_token = read_hf_token(args.hf_token_file)
    if hf_token:
        login(token=hf_token)
    else:
        logging.error("HuggingFace API token is not set.")
        exit(1)

    # Read Mermaid context from file or use default
    if args.mermaid_context_file:
        try:
            with open(args.mermaid_context_file, 'r') as f:
                mermaid_context = f.read()
        except Exception as e:
            logging.error(f"Error reading Mermaid context file '{args.mermaid_context_file}': {e}")
            exit(1)
    else:
        mermaid_context = DEFAULT_MERMAID_PROMPT_TEMPLATE

    directory = args.directory
    summarization_model = args.summarization_model
    mermaid_model = args.mermaid_model
    summarization_tokenizer_name = args.summarization_tokenizer
    mermaid_tokenizer_name = args.mermaid_tokenizer

    # Check if Mermaid file exists
    if os.path.exists(MERMAID_FILE):
        logging.info(f"{MERMAID_FILE} exists. Skipping summarization.")
        # Read mermaid_code from MERMAID_FILE
        with open(MERMAID_FILE, 'r') as f:
            mermaid_code = f.read()
        # Process the Mermaid code
        process_mermaid_diagram(mermaid_code, mermaid_model)
    else:
        # Run summarization
        codebase_summary = summarize_codebase(
            directory,
            summarization_model,
            summarization_tokenizer_name,
        )

        if codebase_summary:
            # Generate Mermaid code
            mermaid_code = generate_mermaid_code(
                codebase_summary,
                mermaid_context,
                mermaid_model,
                mermaid_tokenizer_name
            )
            # Process the Mermaid code
            process_mermaid_diagram(mermaid_code, mermaid_model)
        else:
            logging.warning("No files found or summarized.")
