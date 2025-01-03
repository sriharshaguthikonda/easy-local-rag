import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import nltk
import pynvml
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import numpy as np
import colorama
from colorama import Fore, Back, Style

import pprint

# Set the NLTK data path to include the specific directory
nltk.data.path.append(r"C:\Users\deletable\AppData\Roaming\nltk_data")

# Download the 'punkt_tab' resource if not already available
nltk.download("punkt_tab")

from Semantic_chunking import (
    extract_text_from_html,
    split_into_chunks,
    generate_chunk_id,
)

from Generate_embeddings import check_gpu_temperature


RED = "\033[91m"
RESET_COLOR = "\033[0m"


# Debugging: Print the current working directory and the module path
print("Current working directory:", os.getcwd())
print("Generate_embeddings module path:", os.path.abspath("Generate_embeddings.py"))


"""
TODO : generate_embeddings is the function and not generate_embeddings_for_chunks ....!!!!!!!!

"""
# Import the required functions
import ollama
import json
import chromadb
from chromadb.config import Settings


vault_embeddings = []

# Initialize colorama
colorama.init()


# Enhanced color constants
class LogColors:
    INFO = Fore.CYAN
    DEBUG = Fore.BLUE
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    CRITICAL = Fore.RED + Back.WHITE
    SUCCESS = Fore.GREEN
    PROCESSING = Fore.MAGENTA
    HEADER = Fore.WHITE + Back.BLUE
    RESET = Style.RESET_ALL


class ColoredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create console handler with custom formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def debug(self, msg):
        self.logger.debug(f"{LogColors.DEBUG}{msg}{LogColors.RESET}")

    def info(self, msg):
        self.logger.info(f"{LogColors.INFO}{msg}{LogColors.RESET}")

    def warning(self, msg):
        self.logger.warning(f"{LogColors.WARNING}{msg}{LogColors.RESET}")

    def error(self, msg):
        self.logger.error(f"{LogColors.ERROR}{msg}{LogColors.RESET}")

    def critical(self, msg):
        self.logger.critical(f"{LogColors.CRITICAL}{msg}{LogColors.RESET}")

    def success(self, msg):
        self.logger.info(f"{LogColors.SUCCESS}{msg}{LogColors.RESET}")

    def processing(self, msg):
        self.logger.info(f"{LogColors.PROCESSING}{msg}{LogColors.RESET}")

    def header(self, msg):
        self.logger.info(f"{LogColors.HEADER}{msg}{LogColors.RESET}")


# Initialize logger
logger = ColoredLogger("chunk_processor")

# Replace existing logging configuration
logging.basicConfig(level=logging.DEBUG)

# Dictionary to store the last modification time of files
file_mod_times = {}

# Initialize ChromaDB client
client = chromadb.PersistentClient(settings=Settings())
collection_name = "html_chunks_temp"  # Ensure the collection name matches
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
        results = collection.get(ids=chunk_ids, include=["metadatas"])
        existing_ids = set(results["ids"]) if results else set()
        missing_ids = set(chunk_ids) - existing_ids
        logging.info(f"Existing chunk IDs: {existing_ids}")
        return list(missing_ids)
    except Exception as e:
        logging.error(f"Error checking existing chunks: {e}")
        return chunk_ids


def calculate_chunk_similarity(chunk1, chunk2):
    """Calculate similarity between two chunks using SequenceMatcher."""
    return SequenceMatcher(None, chunk1, chunk2).ratio()


def find_matching_chunks(new_chunk, existing_chunks, similarity_threshold=0.8):
    """Find matching chunks from existing chunks based on similarity."""
    matches = []
    for existing_chunk in existing_chunks:
        similarity = calculate_chunk_similarity(
            new_chunk["text"], existing_chunk["text"]
        )
        if similarity >= similarity_threshold:
            matches.append((existing_chunk, similarity))
    return matches


def get_existing_file_chunks(file_path):
    """Retrieve all chunks for a specific file from ChromaDB."""
    results = collection.get(where={"file_name": file_path})
    return [
        {"text": meta["text"], "id": id}
        for meta, id in zip(results["metadatas"], results["ids"])
    ]


@dataclass
class ChunkDiff:
    to_add: List[dict]
    to_remove: Set[str]
    modified: List[Tuple[str, dict]]  # (old_id, new_chunk)


def align_chunks(
    new_chunks: List[dict],
    existing_chunks: List[dict],
    similarity_threshold: float = 0.85,
) -> ChunkDiff:
    """
    Align new chunks with existing chunks and determine required changes.
    Returns chunks to add, remove, and modify.
    """
    to_add = []
    to_remove = set()
    modified = []
    matched_existing = set()

    # First pass: Find exact and close matches
    for new_chunk in new_chunks:
        best_match = None
        best_similarity = 0

        for existing_chunk in existing_chunks:
            if existing_chunk["id"] in matched_existing:
                continue

            similarity = calculate_chunk_similarity(
                new_chunk["text"], existing_chunk["text"]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_chunk

        if best_similarity >= similarity_threshold:
            matched_existing.add(best_match["id"])
            if best_similarity < 0.95:  # Close but not exact match
                modified.append((best_match["id"], new_chunk))
        else:
            to_add.append(new_chunk)

    # Find unmatched existing chunks to remove
    for existing_chunk in existing_chunks:
        if existing_chunk["id"] not in matched_existing:
            to_remove.add(existing_chunk["id"])

    return ChunkDiff(to_add=to_add, to_remove=to_remove, modified=modified)


def sequence_align_chunks(
    new_chunks: List[dict],
    existing_chunks: List[dict],
    gap_penalty: float = -0.1,
    match_threshold: float = 0.7,
) -> List[Tuple[int, int, float]]:
    """
    Implements a modified Smith-Waterman algorithm for chunk alignment.
    Returns list of (new_idx, existing_idx, similarity_score) tuples.
    """
    n = len(new_chunks)
    m = len(existing_chunks)

    # Initialize scoring matrix
    score_matrix = np.zeros((n + 1, m + 1))
    traceback = np.zeros((n + 1, m + 1, 2), dtype=int)

    # Fill the scoring matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            similarity = calculate_chunk_similarity(
                new_chunks[i - 1]["text"], existing_chunks[j - 1]["text"]
            )

            # Calculate possible scores
            match_score = score_matrix[i - 1][j - 1] + similarity
            delete_score = score_matrix[i - 1][j] + gap_penalty
            insert_score = score_matrix[i][j - 1] + gap_penalty

            # Find best score
            score_matrix[i][j] = max(0, match_score, delete_score, insert_score)

            # Store traceback info
            if score_matrix[i][j] == match_score:
                traceback[i][j] = [i - 1, j - 1]
            elif score_matrix[i][j] == delete_score:
                traceback[i][j] = [i - 1, j]
            else:
                traceback[i][j] = [i, j - 1]

    # Find alignments by traceback
    alignments = []
    while True:
        current_max = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
        if score_matrix[current_max] == 0:
            break

        i, j = current_max
        if score_matrix[i][j] >= match_threshold:
            alignments.append((i - 1, j - 1, score_matrix[i][j]))

        # Zero out region to find next best alignment
        score_matrix[i - 1 : i + 2, j - 1 : j + 2] = 0

    return sorted(alignments, key=lambda x: x[2], reverse=True)


def process_aligned_chunks(
    new_chunks: List[dict],
    existing_chunks: List[dict],
    alignments: List[Tuple[int, int, float]],
) -> ChunkDiff:
    """
    Process aligned chunks to determine what needs to be added, removed, or modified.
    """
    used_new = set()
    used_existing = set()
    to_add = []
    to_remove = set()
    modified = []

    # Process alignments in order of similarity score
    for new_idx, existing_idx, score in alignments:
        if new_idx in used_new or existing_idx in used_existing:
            continue

        used_new.add(new_idx)
        used_existing.add(existing_idx)

        if score < 0.95:  # Similar but not identical
            modified.append((existing_chunks[existing_idx]["id"], new_chunks[new_idx]))

    # Add unmatched new chunks
    for i, chunk in enumerate(new_chunks):
        if i not in used_new:
            to_add.append(chunk)

    # Remove unmatched existing chunks
    for i, chunk in enumerate(existing_chunks):
        if i not in used_existing:
            to_remove.add(chunk["id"])

    return ChunkDiff(to_add=to_add, to_remove=to_remove, modified=modified)


def process_file(self, file_path):
    """Process new or modified files with sequence alignment."""
    try:
        logger.header(f"Processing file: {file_path}")
        modification_time = os.path.getmtime(file_path)

        if (
            file_path in file_mod_times
            and file_mod_times[file_path] == modification_time
        ):
            logger.debug(f"File {file_path} hasn't been modified. Skipping.")
            return

        logger.processing("Extracting text from file...")
        text = extract_text_from_html(file_path)
        if not text:
            logger.warning(f"No text extracted from {file_path}. Skipping.")
            return

        logger.processing("Creating chunks...")
        new_chunks = split_into_chunks(text)
        existing_chunks = get_existing_file_chunks(file_path)
        logger.debug(
            f"Found {len(new_chunks)} new chunks and {len(existing_chunks)} existing chunks"
        )

        logger.processing("Performing sequence alignment...")
        alignments = sequence_align_chunks(new_chunks, existing_chunks)
        diff = process_aligned_chunks(new_chunks, existing_chunks, alignments)

        if diff.to_remove:
            logger.warning(f"Removing {len(diff.to_remove)} outdated chunks")
            collection.delete(ids=list(diff.to_remove))
            logger.success(f"Successfully removed {len(diff.to_remove)} chunks")

        chunks_to_embed = diff.to_add + [chunk for _, chunk in diff.modified]

        if chunks_to_embed:
            logger.processing(
                f"Generating embeddings for {len(chunks_to_embed)} chunks..."
            )
            embeddings = []
            processed_chunks = []

            for i, chunk in enumerate(chunks_to_embed, 1):
                logger.debug(f"Processing chunk {i}/{len(chunks_to_embed)}")
                response = ollama.embeddings(
                    model="mxbai-embed-large", prompt=chunk["text"]
                )

                if check_gpu_temperature() > 51:
                    logger.warning(
                        f"{LogColors.WARNING}GPU temperature critical. Pausing...{LogColors.RESET}"
                    )
                    time.sleep(30)
                    continue

                if "embedding" in response:
                    embeddings.append(response["embedding"])
                    processed_chunks.append(chunk)
                    logger.debug(f"Successfully embedded chunk {i}")
                else:
                    logger.error(f"Failed to generate embedding for chunk {i}")

            if processed_chunks:
                logger.processing("Adding chunks to ChromaDB...")
                ids = [generate_chunk_id(chunk["text"]) for chunk in processed_chunks]
                metadatas = [
                    {
                        "text": chunk["text"],
                        "file_name": file_path,
                        "modification_time": modification_time,
                    }
                    for chunk in processed_chunks
                ]

                collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
                logger.success(
                    f"Successfully added {len(ids)} new/modified chunks to ChromaDB"
                )

        file_mod_times[file_path] = modification_time
        logger.success(f"Completed processing file: {file_path}")

    except Exception as e:
        logger.critical(f"Error processing file {file_path}: {str(e)}")


def check_file_modified(file_path, current_time) -> bool:
    """
    Check if file has been modified since last processing.
    Returns True if file needs processing, False otherwise.
    """
    if file_path not in file_mod_times:
        logger.info(f"New file detected: {file_path}")
        return True

    if file_mod_times[file_path] < current_time:
        logger.info(f"File modified: {file_path}")
        logger.debug(f"Previous mod time: {file_mod_times[file_path]}")
        logger.debug(f"Current mod time: {current_time}")
        return True

    logger.debug(f"File {file_path} hasn't been modified. Skipping.")
    return False


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

            if check_file_modified(file_path, current_mod_time):
                logger.processing(f"Processing modified file: {file_path}")
                self.process_file(file_path)
            else:
                logger.debug(f"Ignoring unchanged file: {file_path}")

    def process_file(self, file_path):
        """Process new or modified files with sequence alignment."""
        try:
            modification_time = os.path.getmtime(file_path)

            # Double check modification time before processing
            if not check_file_modified(file_path, modification_time):
                return

            logger.header(f"Processing file: {file_path}")
            # ...rest of the processing code...

            logger.processing("Extracting text from file...")
            text = extract_text_from_html(file_path)
            if not text:
                logger.warning(f"No text extracted from {file_path}. Skipping.")
                return

            logger.processing("Creating chunks...")
            new_chunks = split_into_chunks(text)
            existing_chunks = get_existing_file_chunks(file_path)
            logger.debug(
                f"Found {len(new_chunks)} new chunks and {len(existing_chunks)} existing chunks"
            )

            logger.processing("Performing sequence alignment...")
            alignments = sequence_align_chunks(new_chunks, existing_chunks)
            diff = process_aligned_chunks(new_chunks, existing_chunks, alignments)

            if diff.to_remove:
                logger.warning(f"Removing {len(diff.to_remove)} outdated chunks")
                collection.delete(ids=list(diff.to_remove))
                logger.success(f"Successfully removed {len(diff.to_remove)} chunks")

            chunks_to_embed = diff.to_add + [chunk for _, chunk in diff.modified]

            if chunks_to_embed:
                logger.processing(
                    f"Generating embeddings for {len(chunks_to_embed)} chunks..."
                )
                embeddings = []
                processed_chunks = []

                for i, chunk in enumerate(chunks_to_embed, 1):
                    logger.debug(f"Processing chunk {i}/{len(chunks_to_embed)}")
                    response = ollama.embeddings(
                        model="mxbai-embed-large", prompt=chunk["text"]
                    )

                    if check_gpu_temperature() > 51:
                        logger.warning(
                            f"{LogColors.WARNING}GPU temperature critical. Pausing...{LogColors.RESET}"
                        )
                        time.sleep(30)
                        continue

                    if "embedding" in response:
                        embeddings.append(response["embedding"])
                        processed_chunks.append(chunk)
                        logger.debug(f"Successfully embedded chunk {i}")
                    else:
                        logger.error(f"Failed to generate embedding for chunk {i}")

                if processed_chunks:
                    logger.processing("Adding chunks to ChromaDB...")
                    ids = [
                        generate_chunk_id(chunk["text"]) for chunk in processed_chunks
                    ]
                    metadatas = [
                        {
                            "text": chunk["text"],
                            "file_name": file_path,
                            "modification_time": modification_time,
                        }
                        for chunk in processed_chunks
                    ]

                    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
                    logger.success(
                        f"Successfully added {len(ids)} new/modified chunks to ChromaDB"
                    )

            file_mod_times[file_path] = modification_time
            logger.success(f"Completed processing file: {file_path}")

        except Exception as e:
            logger.critical(f"Error processing file {file_path}: {str(e)}")

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
