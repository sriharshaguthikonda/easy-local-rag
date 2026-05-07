import os
from tkinter import filedialog, Tk
import re
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import hashlib
import nltk.data  # Import NLTK sentence tokenizer

from rag_config import get_optional_path
from vault_store import atomic_write_json, load_vault, merge_vault_entries


"""
TODO: after this u will have to
ollama serve
then you can run

"c:/Users/deletable/OneDrive/Windows_software/openai whisper/openai/Scripts/python.exe" c:/Users/deletable/OneDrive/easy-local-rag/force_GPU_localrag_no_rewrite.py

TODO: vault file should have the full file name and the modification date and if the file is modified then we will update th contents associated with the file or else skip the file

"""

# Initialize NLTK sentence tokenizer
nltk.download("punkt")
sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


# Function to clean text from unwanted characters
def clean_text(text):
    # Replace special characters with a space or remove them as needed
    cleaned_text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    cleaned_text = (
        re.sub(r"\s+", " ", cleaned_text).strip() + " "
    )  # Normalize whitespace
    return cleaned_text


# Function to extract text from a single HTML file
def extract_text_from_html(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as html_file:
            soup = BeautifulSoup(html_file, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            cleaned_text = clean_text(text)
            return cleaned_text
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""


# Function to generate a unique identifier for a chunk
def generate_chunk_id(text):
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()


# Function to split text into chunks based on sentence boundaries
def split_into_chunks(text):
    # Use NLTK sentence tokenizer for more accurate sentence splitting
    sentences = sentence_tokenizer.tokenize(text.strip())
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for space
            current_chunk += (sentence + " ").strip()
        else:
            if current_chunk:
                chunks.append(
                    {
                        "id": generate_chunk_id(current_chunk),
                        "text": current_chunk.strip(),
                    }
                )
            current_chunk = (sentence + " ").strip()
    # Add the last chunk
    if current_chunk:
        chunks.append(
            {
                "id": generate_chunk_id(current_chunk),
                "text": current_chunk.strip(),
            }
        )

    return chunks


# Function to convert HTML files to text and save to JSON
def convert_html_to_json(directory_path):
    vault_path = "vault.json"
    existing_data = load_vault(vault_path)

    html_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith((".htm", ".html", ".xhtml", ".shtml", ".dhtml")):
                html_files.append(os.path.join(root, file))

    # Use multiprocessing to process files in parallel
    with Pool(cpu_count()) as pool:
        all_texts = list(
            tqdm(
                pool.imap(extract_text_from_html, html_files),
                total=len(html_files),
                desc="Processing HTML files",
            )
        )

    new_data = []
    for file_path, text in zip(html_files, all_texts):
        if text:
            modification_time = os.path.getmtime(file_path)
            if not any(
                entry["file_name"] == os.path.normpath(file_path)
                and entry["modification_time"] == modification_time
                for entry in existing_data
            ):
                # Split text into chunks with more sophisticated sentence splitting
                chunks = split_into_chunks(text)

                new_data.append(
                    {
                        "file_name": os.path.normpath(file_path),
                        "modification_time": modification_time,
                        "chunks": chunks,
                    }
                )
            else:
                print(f"Skipping file {file_path} as it is already in the vault.json")

    merged_data = merge_vault_entries(existing_data, new_data)
    atomic_write_json(vault_path, merged_data)

    print(f"Vault updated: {len(new_data)} new/updated entries written to {vault_path}.")


# Main function to handle folder selection and processing
def main():
    configured_path = get_optional_path("EASY_RAG_VAULT_SOURCE_DIR")
    if configured_path and configured_path.exists():
        directory_path = str(configured_path)
    else:
        root = Tk()
        root.withdraw()  # Hide the main window
        directory_path = filedialog.askdirectory()  # Open the folder selection dialog

    if directory_path:
        convert_html_to_json(directory_path)


if __name__ == "__main__":
    main()
