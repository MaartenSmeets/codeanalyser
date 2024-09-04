import os
import shelve
import json
import uuid
import time  # To add delay between retries
from hashlib import md5
from transformers import AutoTokenizer
from ollama import generate
import subprocess
import logging
from datetime import datetime
import traceback
import textwrap
import matplotlib.pyplot as plt
from ollama._types import ResponseError
import chardet

# Configuration: Specify the Ollama model and context length
OLLAMA_MODEL = 'deepseek-coder-v2:16b-lite-instruct-q5_K_M'
TOKENIZER_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
CACHE_DIR = 'llm_cache'
MAX_CONTEXT_LENGTH = 1024  # Maximum token length for a single prompt
SUMMARIES_DIR = "summaries"
MAX_RETRIES = 5  # Configurable retry attempts for HTTP 500 errors
INITIAL_RETRY_DELAY = 5  # Initial delay in seconds for retries

# Configure logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to initialize the shelve cache
def init_cache():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return shelve.open(os.path.join(CACHE_DIR, 'llm_cache.db'))

# Generate a unique hash for cache key based on input
def generate_cache_key(prompt, model):
    key_string = f"{model}_{prompt}"
    return md5(key_string.encode()).hexdigest()

# Function to initialize tokenizer based on the model
def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer

# Function to split text into chunks based on token length
def split_into_chunks(text, max_tokens, tokenizer):
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    chunk_size = max_tokens
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        
    return chunks

# Helper function to generate unique filenames
def generate_unique_filename(base_name, extension):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    return f"{base_name}_{timestamp}_{unique_id}.{extension}"


def retry_with_fix(prompt, model, retry_attempts=MAX_RETRIES):
    attempt = 0
    retry_delay = INITIAL_RETRY_DELAY
    while attempt < retry_attempts:
        try:
            if attempt > 1:
                logging.info(f"Attempt {attempt + 1}/{retry_attempts} for generating response.")
            response = generate(model=model, prompt=prompt)
            
            # Check if the response contains a valid result
            if response.get('response', None):
                logging.info(f"Response received on attempt {attempt + 1}: {response['response'][:200]}...")  # Log the first 200 characters
                return response.get('response', "")
            else:
                raise Exception("No valid response in the response body.")
                
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1}: {e}. Attempting to request a fix from LLM.")
            # Attempt to get a fix from the LLM
            fix_prompt = f"The following error occurred while processing your request:\n\n{str(e)}\n\nCan you suggest a possible fix for this error?"
            fix_response = generate(model=model, prompt=fix_prompt)
            fix_content = fix_response.get('response', "")
            
            if fix_content:
                logging.info(f"Fix provided by LLM: {fix_content}")
                prompt += f"\n# Fix attempt: {fix_content}"  # Appending the fix to the prompt for retry
                time.sleep(retry_delay)
                retry_delay *= 2  # Continue exponential backoff for retry with fix
                attempt += 1
            else:
                logging.error(f"No fix provided by LLM. Exiting retry loop.")
                break

    logging.error(f"All {retry_attempts} attempts failed.")
    return ""


# Function to call the LLM via Ollama to generate summaries (with caching via shelve)
def generate_response_with_ollama(prompt, model=OLLAMA_MODEL):
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
        response_content = retry_with_fix(prompt, model)
        
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
        return ""

# Function to summarize chunks and save the result
def summarize_chunk_summaries(chunk_summaries, file_path, model=OLLAMA_MODEL):
    chunk_summary_text = "\n\n".join(chunk_summaries)
    
    prompt = f"""
    You have just summarized a large file in multiple parts. Now, based on the following chunk summaries, create a single, concise, structured summary of the entire file:
    
    - The filename: {file_path}
    - Combine all relevant details, avoiding duplication.
    - Focus on the most important points such as:
      - Purpose and functionality
      - Key methods or components
      - Dependencies
      - Any relevant insights about data flow, inputs, outputs, etc.

    Here are the chunk summaries:
    {chunk_summary_text}
    """
    
    final_summary = generate_response_with_ollama(prompt, model)
    return final_summary

def is_test_file(file_path):
    # Convert the file path to lowercase for case-insensitive comparison
    file_path_lower = file_path.lower()
    
    # Define common indicators that the file is a test file
    test_indicators = [
        "src/test",        # General test directories
        "test/resources",  # Test resources
        "test/java",       # Test Java code
        "test"             # General test folder pattern
    ]
    
    # Check if any of the test indicators are in the file path
    for indicator in test_indicators:
        if indicator in file_path_lower:
            return True
    
    return False

# Function to generate a summary for each file
def generate_summary(file_path, file_content, model=OLLAMA_MODEL):
    _, file_extension = os.path.splitext(file_path)

    # Check if the file is a test file
    if is_test_file(file_path):
        logging.info(f"Skipping test file: {file_path}")
        return None, True  # Return None for the summary and True to indicate it is a test file

    prompt_template = f"""
    You are summarizing files in a software repository. Provide a concise but complete and detailed english structured summary of this file. When relevant for this file type and function, explicitly mention inputs, outputs, dependencies, message definitions and describe functional data flow through this file. Add additional information if it will be valuable to understand the codebase better.:

    - The filename: {file_path}
    """

    tokenizer = get_tokenizer(TOKENIZER_NAME)
    prompt = f"{prompt_template}\n{file_content}"

    prompt_token_count = len(tokenizer(prompt, return_tensors="pt").input_ids[0])

    if prompt_token_count > MAX_CONTEXT_LENGTH:
        logging.debug(f"File '{file_path}' exceeds context length; processing in chunks.")
        chunks = split_into_chunks(file_content, MAX_CONTEXT_LENGTH - prompt_token_count, tokenizer)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"{prompt_template}\n{chunk}"
            logging.info(f"Processing chunk {i+1}/{len(chunks)} for file '{file_path}'")
            chunk_summary = generate_response_with_ollama(chunk_prompt, model)
            chunk_summaries.append(chunk_summary)

            chunk_file_name = generate_unique_filename(os.path.basename(file_path), f"chunk_{i+1}.txt")
            save_output_to_file(chunk_summary, os.path.join(SUMMARIES_DIR, chunk_file_name))

        final_summary = summarize_chunk_summaries(chunk_summaries, file_path, model)
        return final_summary, False
    else:
        return generate_response_with_ollama(prompt, model), False

# Function to list all files in the directory
def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

# Function to read the contents of a file with robust encoding handling
def read_file(file_path):
    try:
        # Read a small portion of the file to guess the encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)  # Read the first 10 KB for encoding detection
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'

        # Now open the file with the detected encoding
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return ""

# Function to save output to a file
def save_output_to_file(content, file_name):
    with open(file_name, 'w') as f:
        f.write(content)

# Function to summarize the entire repository with progress indication
def summarize_codebase(directory, model=OLLAMA_MODEL):
    all_files = list_all_files(directory)
    total_files = len(all_files)
    codebase_summary = []
    logging.info(f"Total files to process: {total_files}")

    # Ensure summaries directory exists
    if not os.path.exists(SUMMARIES_DIR):
        os.makedirs(SUMMARIES_DIR)

    for idx, file_path in enumerate(all_files, start=1):
        logging.info(f"Processing file {idx}/{total_files}: {file_path}")
        file_content = read_file(file_path)

        # Skip files that cannot be read
        if not file_content:
            logging.warning(f"Skipping unreadable file: {file_path}")
            continue

        summary, is_test_file = generate_summary(file_path, file_content, model)

        if is_test_file:
            logging.info(f"Test file detected and skipped: {file_path}")
            continue

        if summary:
            codebase_summary.append(f"File: {file_path}\n{summary}\n")
            # Save individual file summary
            file_summary_path = generate_unique_filename(os.path.basename(file_path), "summary.txt")
            save_output_to_file(summary, os.path.join(SUMMARIES_DIR, file_summary_path))

        # Log progress every 5 files
        if idx % 5 == 0 or idx == total_files:
            logging.info(f"Progress: {idx}/{total_files} files processed.")

    logging.info("Codebase summarization complete.")
    return "\n".join(codebase_summary)

# Function to use LLM to generate an extensive prompt for Mermaid diagram creation
def generate_mermaid_prompt(summary):
    prompt = f"""
    Create a detailed Mermaid diagram based on the following structured repository summary. 
    The diagram should cover the following:
    - Key functional components of the system and their relationships.
    - How data flows between these components.
    - Highlight logical separations, such as business logic, data handling, configuration, and presentation layers.
    - Emphasize dependencies between components, including external libraries and internal dependencies.
    - Use logical groupings to separate independent parts of the system.

    Repository summary for Mermaid diagram generation:
    {summary}

    Generate the corresponding Mermaid diagram.
    """
    return prompt

# Function to generate the Mermaid diagram using the LLM
def generate_mermaid_diagram_from_summary(code_summary, model=OLLAMA_MODEL):
    # Generate a prompt based on the combined code summary
    prompt = generate_mermaid_prompt(code_summary)
    
    # Use LLM to generate the Mermaid diagram code
    mermaid_code = generate_response_with_ollama(prompt, model)
    return mermaid_code

# Function to save the mermaid diagram as PNG
def save_mermaid_diagram_as_png(diagram_code, output_file="mermaid_diagram.png"):
    wrapped_code = textwrap.fill(diagram_code, width=80)  # Wrap the diagram for better readability
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, wrapped_code, fontsize=12, ha='center', va='center', family='monospace')
    ax.axis('off')
    plt.savefig(output_file, format='png')
    plt.close()

# Main entry point for summarizing the codebase and generating the diagram
if __name__ == "__main__":
    directory = 'repo'

    # Step 1: Generate summaries for each file in the directory
    codebase_summary = summarize_codebase(directory, model=OLLAMA_MODEL)

    # Step 2: Save the final summary to a file
    if codebase_summary:
        logging.info("Final codebase summary generated.")
        summary_file = generate_unique_filename("codebase_summary", "txt")
        save_output_to_file(codebase_summary, summary_file)
    else:
        logging.warning("No files found or summarized.")

    # Step 3: Generate and save the mermaid diagram based on the functional summary
    mermaid_code = generate_mermaid_diagram_from_summary(codebase_summary)
    mermaid_file = generate_unique_filename("mermaid_diagram", "png")
    save_mermaid_diagram_as_png(mermaid_code, mermaid_file)
