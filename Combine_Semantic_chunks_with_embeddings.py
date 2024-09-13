import json
import csv
import os
from threading import Thread
from queue import Queue
from tqdm import tqdm
import winsound

semantic_vault_path = "semantic_vault.json"
embeddings_path = "Embeddings\\PubMedBert_vault_embeddings.txt"
output_csv_path = "Pubmed_combined_data.csv"


# Function to load JSON file
def load_json(file_path, queue):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    queue.put(data)


# Function to load embeddings file
def load_embeddings(file_path, queue):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Loading embeddings"):
        try:
            data = json.loads(line.strip())
            embeddings.update(data)
        except json.JSONDecodeError as e:
            # Print the problematic line and surrounding characters for debugging
            print(f"JSONDecodeError at line {i+1}, position {e.pos}: {e.msg}")
            error_pos = e.pos
            context_start = max(0, error_pos - 20)
            context_end = min(len(line), error_pos + 20)
            print(f"Problematic data around error:\n{line[context_start:context_end]}")
            continue
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

    all_results = []
    for file in tqdm(semantic_vault, desc="Processing semantic vault"):
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


"""
##    ## ##     ## ##     ## ########  ##    ## 
###   ## ##     ## ###   ### ##     ##  ##  ##  
####  ## ##     ## #### #### ##     ##   ####   
## ## ## ##     ## ## ### ## ########     ##    
##  #### ##     ## ##     ## ##           ##    
##   ### ##     ## ##     ## ##           ##    
##    ##  #######  ##     ## ##           ##    


import json
import csv
import os
from threading import Thread
from queue import Queue
from tqdm import tqdm
import winsound
import numpy as np

semantic_vault_path = "semantic_vault.json"
embeddings_path = "Embeddings\\PubMedBert_vault_embeddings.txt"
output_csv_path = "Pubmed_combined_data.csv"


# Function to load JSON file
def load_json(file_path, queue):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    queue.put(data)


# Function to load embeddings file as numpy arrays
def load_embeddings(file_path, queue):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Loading embeddings"):
        try:
            data = json.loads(line.strip())
            # Convert embedding values to NumPy arrays
            for key, value in data.items():
                embeddings[key] = np.array(value)
        except json.JSONDecodeError as e:
            # Print the problematic line and surrounding characters for debugging
            print(f"JSONDecodeError at line {i+1}, position {e.pos}: {e.msg}")
            error_pos = e.pos
            context_start = max(0, error_pos - 20)
            context_end = min(len(line), error_pos + 20)
            print(f"Problematic data around error:\n{line[context_start:context_end]}")
            continue
    queue.put(embeddings)


# Process chunk
def process_chunk(chunk, embeddings):
    id_ = chunk["id"]
    text = chunk["text"]
    embedding = embeddings.get(id_)
    return [id_, text, embedding] if embedding is not None else None


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

    all_results = []
    for file in tqdm(semantic_vault, desc="Processing semantic vault"):
        results = process_file(file, embeddings)
        all_results.extend(results)

    # Write results to CSV
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
                    json.dumps(embedding.tolist()),  # Convert NumPy array to list for JSON serialization
                ]
            )

    # Signal the end of the script
    os.system("echo \a")
    winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms


if __name__ == "__main__":
    main()

"""
