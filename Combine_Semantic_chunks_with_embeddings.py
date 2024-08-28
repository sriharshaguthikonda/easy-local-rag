import json
import csv
import os
from threading import Thread
from queue import Queue
from tqdm import tqdm
import winsound


# Function to load JSON file
def load_json(file_path, queue):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    queue.put(data)


# Function to load embeddings file
def load_embeddings(file_path, queue):
    with open(file_path, "r", encoding="utf-8") as file:
        embeddings = {}
        for line in file:
            data = json.loads(line.strip())
            embeddings.update(data)
    queue.put(embeddings)


# Process chunk
def process_chunk(chunk, embeddings):
    id_ = chunk["id"]
    text = chunk["text"]
    embedding = embeddings.get(id_)
    return [id_, text, embedding] if embedding else None


# Process file
def process_file(file, embeddings):
    results = []
    file_name = file["file_name"]
    modification_time = file["modification_time"]
    for chunk in file["chunks"]:
        result = process_chunk(chunk, embeddings)
        if result:
            results.append(
                [result[0], result[1], file_name, float(modification_time), result[2]]
            )
    return results


def main():
    semantic_vault_path = "semantic_vault.json"
    embeddings_path = "Embeddings\\mxbai-embed-large_vault_embeddings.txt"
    output_csv_path = "combined_data.csv"

    queue = Queue()

    # Create threads for loading JSON and embeddings
    json_thread = Thread(target=load_json, args=(semantic_vault_path, queue))
    embeddings_thread = Thread(target=load_embeddings, args=(embeddings_path, queue))

    # Start threads
    json_thread.start()
    embeddings_thread.start()

    # Wait for threads to finish
    json_thread.join()
    embeddings_thread.join()

    # Retrieve data from queue
    semantic_vault = queue.get()
    embeddings = queue.get()

    # Limit to the first 5 files for testing
    # limited_vault = semantic_vault[:5]

    all_results = []
    for file in tqdm(semantic_vault):
        results = process_file(file, embeddings)
        all_results.extend(results)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["id", "text", "file_name", "modification_time", "embedding"]
        csvwriter.writerow(header)
        for result in all_results:
            id_, text, file_name, modification_time, embedding = result
            csvwriter.writerow(
                [
                    id_,
                    text,
                    file_name,
                    modification_time,
                    json.dumps(embedding),  # Convert embedding to JSON string
                ]
            )

    # Signal the end of the script
    os.system("echo \a")
    winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms


if __name__ == "__main__":
    main()
