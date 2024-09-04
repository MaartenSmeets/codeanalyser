import os
import shelve
import uuid
from hashlib import md5
from transformers import AutoTokenizer
from ollama import generate
import logging
from datetime import datetime
import traceback
import chardet
import shutil
import subprocess

# Configuration: Specify the Ollama model and context length
OLLAMA_MODEL = 'deepseek-coder-v2:16b-lite-instruct-q4_K_M'
TOKENIZER_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
CACHE_DIR = 'llm_cache'
MAX_CONTEXT_LENGTH = 32000  # Maximum token length for a single prompt
SUMMARIES_DIR = "summaries"
UNPROCESSED_DIR = "unprocessed_files"
MERMAID_DIR = "mermaid"
MERMAID_FILE = "codebase_diagram.mmd"
MERMAID_PNG_FILE = "codebase_diagram.png"

# Configure logging with timestamps, writing to a log file that is overwritten on each new run
log_file = 'script_run.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])

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
def generate_response_with_ollama(prompt, model=OLLAMA_MODEL):
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
        return ""

# Function to generate a Mermaid diagram based on the LLM-generated codebase summary
def generate_mermaid_diagram_from_llm_summary(llm_summary, mermaid_file, output_png):
    logging.info("Generating Mermaid diagram based on LLM codebase summary...")

    # Use the LLM summary to create relationships for the Mermaid diagram
    diagram_content = f"""
    graph TD;
    classDef default fill:#f9f,stroke:#333,stroke-width:2px;

    subgraph Codebase_Structure
    {llm_summary}
    end
    """

    # Save the Mermaid diagram to file
    with open(mermaid_file, 'w') as f:
        f.write(diagram_content)

    # Convert the Mermaid diagram to PNG using mermaid-cli
    try:
        subprocess.run(["mmdc", "-i", mermaid_file, "-o", output_png], check=True)
        logging.info(f"Mermaid diagram saved as PNG at {output_png}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate Mermaid PNG: {e}")

# Function to summarize chunks and save the result
def summarize_chunk_summaries(chunk_summaries, file_path, model=OLLAMA_MODEL):
    chunk_summary_text = "\n\n".join(chunk_summaries)

    prompt = f"""
    You have just summarized a large file in multiple parts. Now, based on the following chunk summaries, create a single, specific, and structured summary of the entire file:

    Filename: {file_path}
    Combine all relevant details, ensuring there is no duplication.
    Focus on the key aspects such as:
    - The overall purpose and functionality of the file.
    - Key methods, components, or modules, and their roles within the file.
    - Dependencies, including any external libraries, frameworks, or other files.
    - Any relevant insights into the data flow, such as inputs, outputs, and how data is processed or transformed.
    Only include information that is directly useful for understanding the file's function within the codebase. Avoid including any irrelevant details or assumptions.
    If applicable, describe specific data used, functional interactions, and how this file contributes to the broader functionality of the system.
    Here are the chunk summaries: {chunk_summary_text}
    """

    final_summary = generate_response_with_ollama(prompt, model)
    return final_summary

# Function to summarize the entire repository and generate the Mermaid diagram
def summarize_codebase(directory, model=OLLAMA_MODEL):
    all_files = list_all_files(directory)
    total_files = len(all_files)
    codebase_summary = []
    logging.info(f"Total files to process: {total_files}")

    if not os.path.exists(SUMMARIES_DIR):
        os.makedirs(SUMMARIES_DIR)

    if not os.path.exists(UNPROCESSED_DIR):
        os.makedirs(UNPROCESSED_DIR)

    for idx, file_path in enumerate(all_files, start=1):
        logging.info(f"Processing file {idx}/{total_files}: {file_path}")
        file_content = read_file(file_path, directory, UNPROCESSED_DIR)

        if not file_content:
            logging.warning(f"Skipping unreadable or empty file: {file_path}")
            continue

        summary, is_test_file = generate_summary(file_path, file_content, model)

        if is_test_file:
            logging.info(f"Test file detected and skipped: {file_path}")
            continue

        if summary:
            codebase_summary.append(f"{summary}\n")
            file_summary_path = generate_unique_filename(
                os.path.basename(file_path), "summary.txt")
            save_output_to_file(summary, os.path.join(
                SUMMARIES_DIR, file_summary_path))

        if idx % 5 == 0 or idx == total_files:
            logging.info(f"Progress: {idx}/{total_files} files processed.")

    # Combine all file summaries for the Mermaid diagram
    combined_summary = "\n".join(codebase_summary)
    
    if combined_summary:
        # Save the final codebase summary
        logging.info("Final codebase summary generated.")
        summary_file = generate_unique_filename("codebase_summary", "txt")
        save_output_to_file(combined_summary, summary_file)

        # Ask the LLM to generate the Mermaid diagram from the summary
        llm_mermaid_prompt = f"""
        Based on the following codebase summary, generate a detailed Mermaid diagram that captures:
        - Functional flows
        - Data objects and their relationships
        - Dependencies between modules
        - Architecture and logical units
        - Key components and their interactions
        - Any other relevant structural information

        Codebase Summary:
        {combined_summary}
        """

        # Call LLM to generate a Mermaid diagram structure
        mermaid_diagram_content = generate_response_with_ollama(llm_mermaid_prompt, model)

        # Generate and save the Mermaid diagram as a PNG
        generate_mermaid_diagram_from_llm_summary(mermaid_diagram_content, MERMAID_FILE, MERMAID_PNG_FILE)

    return combined_summary

# Function to generate a summary for each file
def generate_summary(file_path, file_content, model=OLLAMA_MODEL):
    _, file_extension = os.path.splitext(file_path)

    if is_test_file(file_path):
        logging.info(f"Skipping test file: {file_path}")
        return None, True

    if len(file_content.strip()) == 0:
        logging.warning(f"Skipping empty file: {file_path}")
        return None, True

    prompt_template = f"""
    You are summarizing a file in a software repository. 
    Provide only a specific and concise but detailed and complete English structured summary of this file. 
    Only provide relevant information and nothing else. Be specific referring to content of the provided file and not general.
    Do not make assumptions. Avoid redundancy.
    Only when relevant and useful for understanding the function of the file and codebase as a whole, 
    explicitly and completely mention information like inputs, outputs, specific dependencies, 
    specific data used, and describe functional data flow through this file.

    - The filename: {file_path}
    """

    tokenizer = get_tokenizer(TOKENIZER_NAME)
    
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
            You are summarizing a chunk of a larger file. This is chunk {i+1} of {len(chunks)}.
            Summarize the content of this chunk accurately and specifically. Do not make assumptions about the whole file or include unnecessary information.
            
            - The filename: {file_path}
            Chunk Content:
            {chunk}
            """
            logging.info(f"Processing chunk {i+1}/{len(chunks)} for file '{file_path}'")
            
            try:
                chunk_summary = generate_response_with_ollama(chunk_prompt, model)
                chunk_filename = generate_unique_filename(f"{os.path.basename(file_path)}_chunk_{i+1}", "txt")
                save_output_to_file(chunk_summary, os.path.join(SUMMARIES_DIR, chunk_filename))
                chunk_summaries.append(chunk_summary)

            except Exception as e:
                logging.error(f"Error processing chunk {i+1}/{len(chunks)} for file '{file_path}': {e}")
                copy_unreadable_file(file_path, 'repo', UNPROCESSED_DIR)
                return None, True

        final_summary = summarize_chunk_summaries(chunk_summaries, file_path, model)
        return final_summary, False
    else:
        return generate_response_with_ollama(prompt, model), False

# Function to determine if the file is a test file
def is_test_file(file_path):
    file_path_lower = file_path.lower()
    test_indicators = [
        "src/test",
        "test/resources",
        "test/java",
        "test"
    ]
    for indicator in test_indicators:
        if indicator in file_path_lower:
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
            all_files.append(os.path.join(root, file))
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
    directory = 'repo'

    codebase_summary = summarize_codebase(directory, model=OLLAMA_MODEL)

    if codebase_summary:
        logging.info("Final codebase summary generated.")
    else:
        logging.warning("No files found or summarized.")
