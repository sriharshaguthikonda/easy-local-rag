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

import os
import csv
import json
from tqdm import tqdm
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
import winsound


def init_milvus():
    connections.connect("default", host="localhost", port="19530")

    collection_name = "html_chunks"

    id_field = FieldSchema(
        name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024
    )
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=50000)
    file_name_field = FieldSchema(
        name="file_name", dtype=DataType.VARCHAR, max_length=1024
    )
    modification_time_field = FieldSchema(
        name="modification_time", dtype=DataType.FLOAT
    )
    embedding_field = FieldSchema(
        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024
    )  # Adjust dim as per your embeddings

    schema = CollectionSchema(
        fields=[
            id_field,
            text_field,
            file_name_field,
            modification_time_field,
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


def read_csv_in_batches(csv_path, batch_size=2000):
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)

        batch = []
        for row in tqdm(csvreader, desc="Reading CSV"):
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def insert_batches_to_milvus(collection, csv_path, batch_size=2000):
    for batch in read_csv_in_batches(csv_path, batch_size):
        entities = {
            "file_name": [],
            "modification_time": [],
            "id": [],
            "text": [],
            "embedding": [],
        }

        for row in batch:
            chunk_id = row[0]
            text = row[1]
            file_name = row[2]
            modification_time = float(row[3])  # Ensure it's a float
            embedding_str = row[4]
            embedding = json.loads(embedding_str)  # Ensure it's a list of floats

            # print(f"chunk_id: {type(chunk_id)}, text: {type(text)}, file_name: {type(file_name)}, modification_time: {type(modification_time)}, embedding: {type(embedding)}")

            entities["id"].append(chunk_id)
            entities["text"].append(text)
            entities["file_name"].append(file_name)
            entities["modification_time"].append(modification_time)
            entities["embedding"].append(embedding)

        collection.upsert(
            [
                entities["id"],
                entities["text"],
                entities["file_name"],
                entities["modification_time"],
                entities["embedding"],
            ]
        )
        print(f"Inserted {len(batch)} records to Milvus")


def main():
    """
    try:
        # Get the collection you want to delete
        collection = Collection(name="html_chunks")
        # Drop the collection
        collection.drop()
        print("Collection html_chunks deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    """

    csv_path = "combined_data.csv"
    collection = init_milvus()
    insert_batches_to_milvus(collection, csv_path)
    winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms


if __name__ == "__main__":
    main()
