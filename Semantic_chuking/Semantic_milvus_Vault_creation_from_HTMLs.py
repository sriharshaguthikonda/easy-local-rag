import os
import re
from tkinter import filedialog, Tk
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import hashlib
import nltk.data
from semantic_text_splitter import TextSplitter
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
import winsound

# Initialize NLTK sentence tokenizer
nltk.download("punkt")
sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


BATCH_SIZE = 100  # Adjust the batch size according to your needs


# Function to clean text from unwanted characters
def clean_text(text):
    cleaned_text = re.sub(r"[^\x00-\x7F]+", " ", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip() + " "
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


splitter = TextSplitter((200, 250), 10)


# Function to split text into chunks based on sentence boundaries
def split_into_chunks(text):
    chunks = splitter.chunks(text)
    result_chunks = []
    for chunk in chunks:
        result_chunks.append(
            {
                "id": generate_chunk_id(chunk),
                "text": chunk.strip(),
                "serial": 0,
            }
        )
    return result_chunks


def init_milvus():
    connections.connect(
        alias="default",
        host="localhost",
        port="19530",
        max_receive_message_size=2**30,  # Set to 1 GB (adjust as needed)
        max_send_message_size=2**30,  # Set to 1 GB (adjust as needed)
    )

    collection_name = "html_chunks"

    id_field = FieldSchema(
        name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
    )
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    file_name_field = FieldSchema(
        name="file_name", dtype=DataType.VARCHAR, max_length=6553
    )
    modification_time_field = FieldSchema(
        name="modification_time", dtype=DataType.FLOAT
    )
    serial_field = FieldSchema(name="serial", dtype=DataType.INT64)
    embedding_field = FieldSchema(
        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024
    )  # Adjust dim as per your embeddings

    schema = CollectionSchema(
        fields=[
            id_field,
            text_field,
            file_name_field,
            modification_time_field,
            serial_field,
            embedding_field,
        ],
        description="HTML text chunks",
    )

    if utility.has_collection(collection_name):
        existing_collection = Collection(collection_name)
        existing_schema = existing_collection.schema

        if existing_schema != schema:
            raise ValueError(
                "The existing collection schema does not match the provided schema."
            )

        return existing_collection
    else:
        collection = Collection(name=collection_name, schema=schema)
        return collection


# Function to convert HTML files to text and save to Milvus
def convert_html_to_milvus(directory_path):
    collection = init_milvus()

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

    entities = {
        "id": [],
        "text": [],
        "file_name": [],
        "modification_time": [],
        "serial": [],
        "embedding": [],
    }

    dummy_embedding = [0.0] * 1024  # Dummy embedding with dimension 1024

    for file_path, text in tqdm(
        zip(html_files, all_texts),
        total=len(html_files),
        desc="Preparing data for Milvus",
    ):
        if text:
            modification_time = os.path.getmtime(file_path)
            chunks = split_into_chunks(text)
            for chunk in chunks:
                entities["id"].append(chunk["id"])
                entities["text"].append(chunk["text"])
                entities["file_name"].append(os.path.normpath(file_path))
                entities["modification_time"].append(modification_time)
                entities["serial"].append(chunk["serial"])
                entities["embedding"].append(dummy_embedding)

                # Check if we have reached the batch size
                if len(entities["id"]) >= BATCH_SIZE:
                    data = [
                        entities["id"],
                        entities["text"],
                        entities["file_name"],
                        entities["modification_time"],
                        entities["serial"],
                        entities["embedding"],
                    ]
                    entities = {
                        "id": [],
                        "text": [],
                        "file_name": [],
                        "modification_time": [],
                        "serial": [],
                        "embedding": [],
                    }
                    collection.insert(data)

    # Insert any remaining data
    if entities["id"]:
        data = [
            entities["id"],
            entities["text"],
            entities["file_name"],
            entities["modification_time"],
            entities["serial"],
            entities["embedding"],
        ]
        collection.insert(data)

    print(f"Inserted all chunks into Milvus with dummy embeddings")

    winsound.Beep(1000, 500)  # Signal the end of the script


# Main function to handle folder selection and processing
def main():
    root = Tk()
    root.withdraw()  # Hide the main window
    directory_path = filedialog.askdirectory()  # Open the folder selection dialog
    if directory_path:
        convert_html_to_milvus(directory_path)


if __name__ == "__main__":
    main()
