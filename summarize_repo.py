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
OLLAMA_MODEL = 'gemma2:9b-instruct-q8_0'
TOKENIZER_NAME = "google/gemma-2-9b-it"
CACHE_DIR = 'llm_cache'
MAX_CONTEXT_LENGTH = 8192  # Maximum token length for a single prompt
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
        raise e  # Re-raise the exception to allow the calling function to handle it.


# Function to summarize chunks and save the result
def summarize_chunk_summaries(chunk_summaries, file_path, model=OLLAMA_MODEL):
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

    final_summary = generate_response_with_ollama(prompt, model)
    return clean_generated_summary(final_summary)


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

        try:
            summary, is_test_file = generate_summary(file_path, file_content, model)

            if is_test_file:
                logging.info(f"Test file detected and skipped: {file_path}")
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
        except Exception:
            logging.error(f"Error processing file: {file_path}")
            # Copy the file to unprocessed_files if it fails
            copy_unreadable_file(file_path, directory, UNPROCESSED_DIR)

    # Combine all file summaries for the Mermaid diagram
    combined_summary = "\n".join(codebase_summary)
    
    if combined_summary:
        # Save the final codebase summary
        logging.info("Final codebase summary generated.")
        summary_file = generate_unique_filename("codebase_summary", "txt")
        save_output_to_file(combined_summary, summary_file)

        # Ask the LLM to generate the Mermaid diagram from the summary
        llm_mermaid_prompt = f"""Based on the comprehensive summary of the entire codebase below, generate a detailed and insightful Mermaid diagram that focuses on the **overall architecture** and **functional flow** of the system. This diagram should offer a high-level, conceptual view of how the different components and modules work together, emphasizing the architectural design and system functionality rather than the granular details of individual files. Ensure the diagram captures the following key aspects:

        1. **System Architecture**: Clearly represent the high-level architecture of the codebase, including core modules, services, and subsystems. Highlight how these components are organized and how they interact to form the overall system.
        - Focus on how different parts of the codebase contribute to the main system functionality.
        - Show the separation of concerns between major system components, subsystems, or layers (e.g., application logic, data access, service integration).

        2. **Functional Relationships**: Illustrate how the system's core functionalities are distributed across different modules or services. Depict the key functional units, their roles, and how they communicate with each other.
        - Show how data, control, and execution flow between these units.
        - Highlight any major processes or workflows that span across multiple components or layers.

        3. **Key Interactions and Dependencies**: Show the key interactions between the system’s components and any important dependencies (both internal and external). This includes:
        - Interactions between major system modules (e.g., service layers, APIs, database access).
        - External dependencies such as third-party libraries, APIs, or services that are critical to the system’s operation.

        4. **Data Flow and Processes**: Visualize how data is handled, processed, and transformed across the system. Show the flow of information from inputs to outputs, and any key transformation or decision points.
        - Focus on critical data processing pathways or pipelines.
        - Indicate how data is exchanged between subsystems or modules, including key storage and retrieval mechanisms if applicable.

        5. **Major Components and Responsibilities**: Highlight the primary components or services within the system, and succinctly represent their responsibilities within the larger architecture.
        - Identify core system functionalities that are handled by specific components (e.g., data processing, user authentication, communication with external services).

        6. **Logical Grouping of Components**: Group related components or services into logical clusters that reflect how they function together within the broader architecture. This could include layers (e.g., user interface, business logic, data storage), services, or microservices.

        7. **Architectural Patterns**: If applicable, showcase any evident architectural patterns, such as client-server models, event-driven architectures, or microservices. Highlight how these patterns influence the structure and interaction of the system’s components.

        8. **System Overview**: Provide a holistic view of the system’s architecture, allowing the viewer to understand the main functions and interactions at a glance. Avoid technical details like file names or code-specific structures unless they are critical to understanding the overall design.

        Make sure to format the output using valid Mermaid syntax, ensuring clarity and readability. Focus on illustrating the **functional and architectural relationships** between components, processes, and data flow, while maintaining an emphasis on how these elements collectively support the system’s overall purpose.

        **Codebase Summary**:
        {combined_summary}
        """

        # Call LLM to generate a Mermaid diagram structure
        mermaid_diagram_content = generate_response_with_ollama(llm_mermaid_prompt, model)

        # Save the Mermaid diagram content to a file
        with open(MERMAID_FILE, 'w') as f:
            f.write(mermaid_diagram_content)

        # Convert the Mermaid diagram to PNG using mermaid-cli
        try:
            subprocess.run(["mmdc", "-i", MERMAID_FILE, "-o", MERMAID_PNG_FILE], check=True)
            logging.info(f"Mermaid diagram saved as PNG at {MERMAID_PNG_FILE}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate Mermaid PNG: {e}")

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

    prompt_template = f"""You are tasked with summarizing a file from a software repository. Provide a **precise**, **comprehensive**, and **well-structured** English summary that accurately reflects the contents of the file. Do not ask for confirmation. Do not provide suggestions. Do not suggest or ask additional input or follow-up questions. Do not indicate potential uses or interpretations. Do not provide recommendations. Focus solely to creating the summary and focus strictly on what is necessary to keep the summary concise. Avoid redundency and do not summarize the summary. The summary must be:

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
            You are summarizing a portion of a file in a software repository. This portion (or chunk) belongs to a larger file, and it is part {i+1} of {len(chunks)}. Summarize strictly and solely the content of this chunk in a clear, structured, and concise manner, focusing on relevant technical and functional details. You are not allowed to ask or suggest follow-up questions or be polite or confirm. Do not summarize the summary. Do not indicate what is missing. Do not indicate potential uses or interpretations. Do not provide recommendations.

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
            - The filename: {file_path}
            **Chunk content**:
            {chunk}
            """

            logging.info(f"Processing chunk {i+1}/{len(chunks)} for file '{file_path}'")
            
            try:
                chunk_summary = generate_response_with_ollama(chunk_prompt, model)
                cleaned_chunk_summary = clean_generated_summary(chunk_summary)
                chunk_filename = generate_unique_filename(f"{os.path.basename(file_path)}_chunk_{i+1}", "txt")
                save_output_to_file(cleaned_chunk_summary, os.path.join(SUMMARIES_DIR, chunk_filename))
                chunk_summaries.append(cleaned_chunk_summary)

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
