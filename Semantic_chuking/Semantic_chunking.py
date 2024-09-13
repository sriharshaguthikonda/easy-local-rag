import os
from tkinter import filedialog, Tk
import re
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import hashlib
import nltk.data  # Import NLTK sentence tokenizer
from semantic_text_splitter import TextSplitter

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


splitter = TextSplitter((800, 900))


# Function to split text into chunks based on sentence boundaries
def split_into_chunks(text):
    chunks = splitter.chunks(text)
    result_chunks = []
    for chunk in chunks:
        result_chunks.append(
            {
                "id": generate_chunk_id(chunk),
                "text": chunk.strip(),
            }
        )
    return result_chunks


# Function to convert HTML files to text and save to JSON
def convert_html_to_json(directory_path):
    # Load existing data to check for duplicates
    existing_data = []
    if (
        os.path.exists("semantic_vault.json")
        and os.path.getsize("semantic_vault.json") > 0
    ):
        try:
            with open("semantic_vault.json", "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Error loading existing data from semantic_vault.json: {e}")
            existing_data = []

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
            key = (os.path.normpath(file_path), modification_time)
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
                print(
                    f"Skipping file {file_path} as it is already in the semantic_vault.json"
                )

    # Write new data to semantic_vault.json
    mode = "a" if os.path.exists("semantic_vault.json") else "w"
    with open("semantic_vault.json", mode, encoding="utf-8") as json_file:
        if (
            os.path.exists("semantic_vault.json")
            and os.path.getsize("semantic_vault.json") > 0
        ):
            json_file.write(",\n")
        json.dump(new_data, json_file, indent=2)

    # Write new data to semantic_vault.json
    mode = "a" if os.path.exists("Backup_semantic_vault.json") else "w"
    with open("semantic_vault.json", mode, encoding="utf-8") as json_file:
        if (
            os.path.exists("semantic_vault.json")
            and os.path.getsize("semantic_vault.json") > 0
        ):
            json_file.write(",\n")
        json.dump(new_data, json_file, indent=2)

    print(
        f"HTML files content appended to semantic_vault.json with each entry on a separate line."
    )


# Main function to handle folder selection and processing
def main():
    root = Tk()
    root.withdraw()  # Hide the main window
    directory_path = filedialog.askdirectory()  # Open the folder selection dialog
    if directory_path:
        convert_html_to_json(directory_path)


if __name__ == "__main__":
    main()
