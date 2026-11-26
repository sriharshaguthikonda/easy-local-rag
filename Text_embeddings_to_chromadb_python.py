"""
TODO: Milvus does not currently have native support for running directly on Windows without using Docker or WSL. The Milvus development team has indicated that they do not plan to add support for Windows directly at this time【9†source】【10†source】.

If you want to install Milvus without Docker, you would typically need to use a Linux environment. However, if you want to run Milvus natively on Windows, your best option is to use Windows Subsystem for Linux (WSL) to create a Linux environment within Windows【9†source】【10†source】.

Here are some general steps if you choose to use WSL:

1. **Install WSL**: Enable WSL and install a Linux distribution from the Microsoft Store.

2. **Install Milvus Dependencies**: Use the Linux terminal in WSL to install necessary dependencies like `etcd` and `MinIO`.

3. **Install Milvus**: Follow the [Milvus documentation](https://milvus.io/docs/install_standalone-docker.md) for Linux installation, modifying any steps as necessary for the WSL environment.

4. **Run Milvus**: Once everything is set up, you can run Milvus using the provided scripts in the distribution.

Alternatively, you can continue using Docker on Windows as a workaround until native support becomes available.
"""

import csv
import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import winsound

from embedding_contract import (
    DEFAULT_EMBEDDING_MODEL,
    open_collection_for_embedding_write,
    validate_embedding_compatibility,
)
"""TODO 1: Update the `csv_path` variable to point to the location of your processed data CSV file."""
"""TODO 1: Update the `csv_path` variable to point to the location of your processed data CSV file."""
"""TODO 1: Update the `csv_path` variable to point to the location of your processed data CSV file."""


csv_path = "processed_data.csv"
collection_name = "html_chunks_text_in_documents"


def init_chromadb():
    client = chromadb.PersistentClient(
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    return open_collection_for_embedding_write(client, collection_name, DEFAULT_EMBEDDING_MODEL)


def _load_provenance(path):
    manifest_path = path.with_name(path.name + ".provenance.json")
    try:
        with manifest_path.open(encoding="utf-8") as manifest:
            provenance = json.load(manifest)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read CSV provenance: {manifest_path.name}.") from error

    version = provenance.get("schema_version") if isinstance(provenance, dict) else None
    if type(version) is not int or version != 1:
        raise ValueError("CSV provenance must have schema_version 1.")
    if provenance.get("embedding_model") != DEFAULT_EMBEDDING_MODEL:
        raise ValueError("CSV provenance embedding_model does not match the configured writer.")
    dimensions = provenance.get("dimensions")
    if type(dimensions) is not int or dimensions <= 0:
        raise ValueError("CSV provenance dimensions must be a positive integer.")
    digest = provenance.get("csv_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("CSV provenance requires a lowercase SHA-256 digest.")
    return provenance


def _read_csv_rows(path):
    with path.open(encoding="utf-8", newline="") as csvfile:
        reader = csv.reader(csvfile, strict=True)
        header = next(reader, None)
        if header != ["id", "text", "file_name", "modification_time", "embedding"]:
            raise ValueError("CSV header must match the five-column vector export format.")
        yield from reader


def _finite_vector_value(value):
    # Chroma stores vectors as float32.
    return type(value) in (int, float) and abs(value) <= 3.4028234663852886e38


def _validate_csv_rows(snapshot, dimensions):
    for row_number, row in enumerate(_read_csv_rows(snapshot), start=2):
        try:
            if len(row) != 5 or not row[0].strip():
                raise ValueError
            timestamp = float(row[3])
            vector = json.loads(row[4])
            if (
                not math.isfinite(timestamp)
                or not isinstance(vector, list)
                or len(vector) != dimensions
                or not all(_finite_vector_value(value) for value in vector)
            ):
                raise ValueError
        except ValueError as error:
            raise ValueError(f"CSV row {row_number} has invalid fields or embedding.") from error


@contextmanager
def _validated_csv_snapshot(csv_path):
    path = Path(csv_path)
    provenance = _load_provenance(path)
    with TemporaryDirectory(prefix="rag-csv-") as directory:
        snapshot = Path(directory) / "input.csv"
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source, snapshot.open("wb") as copy:
                remaining = os.fstat(source.fileno()).st_size
                while block := source.read(1024 * 1024):
                    remaining -= len(block)
                    if remaining < 0:
                        break
                    copy.write(block)
                    digest.update(block)
                if remaining != 0:
                    raise ValueError("CSV size changed while taking its snapshot.")
        except OSError as error:
            raise ValueError("Could not snapshot the CSV.") from error
        if digest.hexdigest() != provenance["csv_sha256"]:
            raise ValueError("CSV snapshot does not match the provenance SHA-256 digest.")

        try:
            _validate_csv_rows(snapshot, provenance["dimensions"])
        except (csv.Error, OSError) as error:
            raise ValueError("Could not read the CSV snapshot.") from error
        yield snapshot, provenance["embedding_model"]


def read_csv_in_batches(csv_path, batch_size=5460):
    batch = []
    for row in tqdm(_read_csv_rows(Path(csv_path)), desc="Reading CSV"):
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def insert_batches_to_chromadb(collection, csv_path, batch_size=5460):
    """Import only CSVs accompanied by a verified producer provenance sidecar.

    Passing None opens the configured destination after complete preflight.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("CSV batch size must be a positive integer.")
    with _validated_csv_snapshot(csv_path) as (snapshot, model):
        if collection is None:
            collection = init_chromadb()
        validate_embedding_compatibility(collection, model, for_write=True)
        _write_csv_batches(collection, snapshot, batch_size)


def _write_csv_batches(collection, csv_path, batch_size):
    for batch in read_csv_in_batches(csv_path, batch_size):
        ids = []
        metadatas = []
        embeddings = []
        documents = []
        unique_ids = set()

        for row in batch:
            chunk_id = row[0]
            if chunk_id in unique_ids:
                print(f"Duplicate ID found: {chunk_id}, skipping this entry.")
                continue

            unique_ids.add(chunk_id)
            text = row[1]
            file_name = row[2]
            modification_time = float(row[3])  # Ensure it's a float
            embedding_str = row[4]
            embedding = json.loads(embedding_str)  # Ensure it's a list of floats

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(
                {
                    "file_name": file_name,
                    "modification_time": modification_time,
                }
            )
            embeddings.append(embedding)

        try:
            collection.add(
                embeddings=embeddings, metadatas=metadatas, ids=ids, documents=documents
            )
            print(f"Inserted {len(ids)} records to ChromaDB")
        except Exception as e:
            print(f"An error occurred while adding documents: {e}")


def main():
    insert_batches_to_chromadb(None, csv_path)
    winsound.Beep(1000, 500)


if __name__ == "__main__":
    main()


"""
import os
import csv
import json
from tqdm import tqdm
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import winsound
import threading
from queue import Queue

collection_name = "html_chunks"

def preprocess_csv_in_batches(csv_path, queue, batch_size):
    unique_ids = set()
    duplicate_counter = {}
    batch = []

    with open(csv_path, "r", encoding="utf-8") as infile:
        csvreader = csv.reader(infile)
        header = next(csvreader)  # Read and discard the header

        for row in tqdm(csvreader, desc="Preprocessing CSV"):
            chunk_id = row[0]
            text = row[1]
            file_name = row[2]
            modification_time = row[3]  # Keep it as a string for now
            embedding_str = row[4]

            if chunk_id in unique_ids:
                if duplicate_counter.get(chunk_id) is None:
                    duplicate_counter[chunk_id] = [(text, file_name, modification_time, embedding_str)]
                else:
                    found = False
                    for stored_text, stored_file_name, stored_mod_time, stored_embedding in duplicate_counter[chunk_id]:
                        if stored_text == text:
                            found = True
                            break

                    if not found:
                        serial_number = len(duplicate_counter[chunk_id]) + 1
                        new_chunk_id = f"{chunk_id}_{serial_number}"
                        row[0] = new_chunk_id  # Update the chunk ID in the row
                        duplicate_counter[chunk_id].append((text, file_name, modification_time, embedding_str))
            else:
                unique_ids.add(chunk_id)
                duplicate_counter[chunk_id] = [(text, file_name, modification_time, embedding_str)]

            batch.append(row)
            if len(batch) == batch_size:
                queue.put(batch)
                batch = []

        if batch:
            queue.put(batch)

    queue.put(None)  # Signal that preprocessing is done


def read_csv_in_batches(queue, batch_size, processing_queue):
    current_batch = []
    while True:
        rows = queue.get()
        if rows == None:
            break  # Signal the end of preprocessing
        current_batch.extend(rows)

        while len(current_batch) >= batch_size:
            batch_to_insert = current_batch[:batch_size]
            processing_queue.put(batch_to_insert)
            current_batch = current_batch[batch_size:]

    if current_batch:
        processing_queue.put(current_batch)

    processing_queue.put(None)  # Signal that batch reading is done


def insert_batches_to_chromadb(collection, processing_queue):
    while True:
        batch = processing_queue.get()
        if batch == None:
            break  # Stop when batch reader signals completion

        ids = []
        metadatas = []
        embeddings = []
        documents = []
        unique_ids = set()

        for row in batch:
            chunk_id = row[0]
            if chunk_id in unique_ids:
                print(f"Duplicate ID found: {chunk_id}, skipping this entry.")
                continue

            unique_ids.add(chunk_id)
            text = row[1]
            file_name = row[2]
            modification_time = float(row[3])  # Ensure it's a float
            embedding_str = row[4]
            embedding = json.loads(embedding_str)  # Ensure it's a list of floats

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append({
                "file_name": file_name,
                "modification_time": modification_time,
            })
            embeddings.append(embedding)

        try:
            collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids, documents=documents)
            print(f"Inserted {len(ids)} records to ChromaDB")
        except Exception as e:
            print(f"An error occurred while adding documents: {e}")
        finally:
            processing_queue.task_done()


def run_multithreaded_process(csv_path, collection, batch_size=5460, queue_size=10):
    preprocess_queue = Queue(maxsize=queue_size)
    processing_queue = Queue(maxsize=queue_size)

    # Create threads for preprocessing, reading in batches, and inserting into ChromaDB
    preprocess_thread = threading.Thread(target=preprocess_csv_in_batches, args=(csv_path, preprocess_queue, batch_size))
    reader_thread = threading.Thread(target=read_csv_in_batches, args=(preprocess_queue, batch_size, processing_queue))
    inserter_thread = threading.Thread(target=insert_batches_to_chromadb, args=(collection, processing_queue))

    # Start threads
    preprocess_thread.start()
    reader_thread.start()
    inserter_thread.start()

    # Wait for threads to finish
    preprocess_thread.join()
    reader_thread.join()
    inserter_thread.join()

    print("Completed processing")


def init_chromadb():
    client = chromadb.PersistentClient(
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    try:
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        collection = None

    return collection


def main():
    csv_path = "combined_data.csv"
    collection = init_chromadb()
    if collection:
        run_multithreaded_process(csv_path, collection, batch_size=5460)
        winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms


if __name__ == "__main__":
    main()
"""
