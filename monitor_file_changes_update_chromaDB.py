import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import nltk


from Semantic_chunking import (
    extract_text_from_html,
    split_into_chunks,
    generate_chunk_id,
)

from Generate_embeddings import check_gpu_temperature


# Import the required functions
import ollama
import json
import chromadb
from chromadb.config import Settings


# Set the NLTK data path to include the specific directory
nltk.data.path.append(r"C:\Users\deletable\AppData\Roaming\nltk_data")

# Download the 'punkt_tab' resource if not already available
nltk.download("punkt_tab")


RED = "\033[91m"
RESET_COLOR = "\033[0m"


# Debugging: Print the current working directory and the module path
print("Current working directory:", os.getcwd())
print("Generate_embeddings module path:", os.path.abspath("Generate_embeddings.py"))


vault_embeddings = []

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Dictionary to store the last modification time of files
file_mod_times = {}

# Initialize ChromaDB client
client = chromadb.PersistentClient(settings=Settings())


"""TODO :  the file links are not working. it will switch to new chat when i click on the link of the file or open path."""
"""TODO :  the file links are not working. it will switch to new chat when i click on the link of the file or open path."""
"""TODO :  the file links are not working. it will switch to new chat when i click on the link of the file or open path."""


collection_name = "html_chunks_text_in_documents"  # Ensure the collection name matches
collection = client.get_or_create_collection(name=collection_name)


def is_file_in_chromadb(file_path, modification_time):
    """Check if the file is already in ChromaDB."""

    results = collection.get(where={"file_name": file_path})
    print(len(results["metadatas"]))
    # pprint.pprint(results)  # Debugging: Print the results to inspect the structure

    return len(results["metadatas"]) > 1


def get_existing_chunk_hashes(file_path, modification_time):
    """Get existing chunk hashes from ChromaDB for a specific file."""

    results = collection.get(where={"file_name": file_path})
    # pprint.pprint(results)  # Debugging: Print the results to inspect the structure

    return {result["id"] for result in results["file_name"]}


def check_existing_chunks(collection, chunk_ids):
    """Check which chunk IDs are not present in the database."""
    try:
        chunk_ids = list(set(chunk_ids))  # Deduplicate chunk_ids
        results = collection.get(ids=chunk_ids, include=["metadatas"])
        existing_ids = set(results["ids"]) if results else set()
        missing_ids = set(chunk_ids) - existing_ids
        logging.info(f"Existing chunk IDs: {existing_ids}")
        return list(missing_ids)
    except Exception as e:
        logging.error(f"Error checking existing chunks: {e}")
        return chunk_ids


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, folder_path, output_json="temp_vault.json"):
        self.folder_path = folder_path
        self.output_json = output_json

        # Load existing data from temp_vault.json
        if os.path.exists(self.output_json) and os.path.getsize(self.output_json) > 0:
            with open(self.output_json, "r", encoding="utf-8") as f:
                try:
                    self.existing_data = json.load(f)
                except json.JSONDecodeError:
                    self.existing_data = []
        else:
            self.existing_data = []

    def on_created(self, event):
        """Triggered when a new file is created."""
        if not event.is_directory and event.src_path.endswith(
            (".html", ".htm", ".xhtml", ".shtml", ".dhtml")
        ):
            logging.info(f"New file detected: {event.src_path}")
            self.process_file(event.src_path)

    def on_modified(self, event):
        """Triggered when an existing file is modified."""
        if not event.is_directory and event.src_path.endswith(
            (".html", ".htm", ".xhtml", ".shtml", ".dhtml")
        ):
            file_path = event.src_path
            current_mod_time = os.path.getmtime(file_path)
            if (
                file_path not in file_mod_times
                or file_mod_times[file_path] != current_mod_time
            ):
                logging.info(f"File modified: {file_path}")
                self.process_file(file_path)

    def process_file(self, file_path):
        """Process new or modified files."""
        try:
            modification_time = os.path.getmtime(file_path)

            # Extract text and create chunks
            text = extract_text_from_html(file_path)
            if not text:
                logging.warning(f"No text extracted from {file_path}. Skipping.")
                return

            chunks = split_into_chunks(text)

            # Remove duplicates while preserving order
            seen_ids = {}
            unique_chunks = []

            for chunk in chunks:
                chunk_id = generate_chunk_id(chunk["text"])
                if chunk_id not in seen_ids:
                    seen_ids[chunk_id] = True
                    unique_chunks.append(chunk)
                else:
                    logging.info(
                        f"Duplicate chunk detected and removed: {chunk_id[:8]}..."
                    )

            chunks = unique_chunks
            chunk_ids = list(seen_ids.keys())

            # Check which chunks are missing from the database
            missing_chunk_ids = check_existing_chunks(collection, chunk_ids)

            if not missing_chunk_ids:
                logging.info(
                    f"All chunks already exist in database for file: {file_path}"
                )
                return

            # Filter chunks to only process missing ones
            new_chunks = [
                chunk
                for chunk in chunks
                if generate_chunk_id(chunk["text"]) in missing_chunk_ids
            ]

            # Generate embeddings only for new chunks
            embeddings = []
            processed_chunks = []

            for chunk in new_chunks:
                response = ollama.embeddings(
                    model="mxbai-embed-large", prompt=chunk["text"]
                )
                
                # Check GPU temperature if available
                gpu_temp = check_gpu_temperature()
                if gpu_temp is not None and gpu_temp > 51:
                    print(
                        f"{RED}GPU temp is too high ({gpu_temp}°C). Pausing temporarily...{RESET_COLOR}\n"
                    )
                    time.sleep(30)

                if "embedding" in response:
                    embeddings.append(response["embedding"])
                    processed_chunks.append(chunk)
                else:
                    logging.warning(
                        f"Failed to get embedding for chunk: {chunk['text']}"
                    )

            # Add new chunks to ChromaDB
            if processed_chunks:
                ids = [generate_chunk_id(chunk["text"]) for chunk in processed_chunks]
                documents = [chunk["text"] for chunk in processed_chunks]
                metadatas = [
                    {
                        "file_name": file_path,
                        "modification_time": modification_time,
                    }
                    for chunk in processed_chunks
                ]
                collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                logging.info(f"Added {len(ids)} new chunks to ChromaDB")

                # Update vault.json
                new_entry = {
                    "file_name": file_path,
                    "modification_time": modification_time,
                    "chunks": processed_chunks,
                }
                self._update_vault_json(new_entry)

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

    def _update_vault_json(self, new_entry):
        """Helper method to update vault.json"""
        if not any(
            entry["file_name"] == new_entry["file_name"]
            and entry["modification_time"] == new_entry["modification_time"]
            for entry in self.existing_data
        ):
            self.existing_data.append(new_entry)
            file_mod_times[new_entry["file_name"]] = new_entry["modification_time"]
            with open(self.output_json, "w", encoding="utf-8") as f:
                json.dump(self.existing_data, f, indent=2)


def scan_initial_files(folder_path, handler):
    """Initial scan to detect and process all existing files."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".html", ".htm", ".xhtml", ".shtml", ".dhtml")):
                file_path = os.path.join(root, file)
                handler.process_file(file_path)
    logging.info("Initial scan complete. Monitoring for changes...")


def monitor_folder(folder_path):
    """Sets up the folder monitoring."""
    logging.info(f"Starting to monitor folder: {folder_path}")
    handler = FileChangeHandler(folder_path)
    scan_initial_files(folder_path, handler)

    observer = Observer()
    observer.schedule(handler, folder_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Monitoring stopped.")
    observer.join()


if __name__ == "__main__":
    # Specify the folder you want to monitor
    FOLDER_TO_MONITOR = r"C:\Users\deletable\Google Drive"

    # Start monitoring the folder
    monitor_folder(FOLDER_TO_MONITOR)

    # Beep to indicate the script has ended
    import winsound

    winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
