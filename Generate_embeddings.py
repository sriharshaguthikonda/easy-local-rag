import os
import json
import pynvml
import time
from tqdm import tqdm
import ollama
import torch
import threading


RED = "\033[91m"
RESET_COLOR = "\033[0m"

# Constants
EMBEDDINGS_DIR = "Embeddings"
MOD_TIME_FILE = os.path.join(EMBEDDINGS_DIR, "mod_times.json")

PT_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "vault_embeddings.pt")
TXT_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "vault_embeddings.txt")

UPDATED_VAULT_JSON_FILE = "vault.json"


def convert_pt_to_txt(pt_file, txt_file):
    embeddings = torch.load(pt_file)
    with open(txt_file, "w", encoding="utf-8") as f:
        for embedding in embeddings:
            f.write(json.dumps(embedding) + "\n")


def convert_txt_to_pt(txt_file, pt_file):
    embeddings = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            embeddings.append(data)
    torch.save(embeddings, pt_file)


# Function to read vault data from JSON file
def read_vault_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


# Function to save embeddings
def save_embeddings(new_embeddings):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    embeddings_file = os.path.join(EMBEDDINGS_DIR, PT_EMBEDDINGS_FILE)
    # Load existing embeddings if they exist
    if os.path.exists(embeddings_file):
        existing_embeddings = torch.load(embeddings_file)
        existing_embeddings.extend(new_embeddings)
        updated_embeddings = existing_embeddings
    else:
        updated_embeddings = new_embeddings

    # Save the updated embeddings
    torch.save(updated_embeddings, embeddings_file)


# Function to load embeddings and file modification time
def load_embeddings(vault_name):
    embeddings_file = os.path.join(EMBEDDINGS_DIR, f"{vault_name}_embeddings.pt")
    if os.path.exists(embeddings_file) and os.path.exists(MOD_TIME_FILE):
        embeddings = torch.load(embeddings_file)
        # with open(MOD_TIME_FILE, "r") as f:
        # mod_time_data = json.load(f)
        # mod_time = mod_time_data.get(vault_name)
        return embeddings  # , mod_time
    return None, None


# Function to save embeddings in text format
def save_embeddings_txt(new_embeddings):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    embeddings_file = os.path.join(EMBEDDINGS_DIR, "vault_embeddings.txt")

    with open(embeddings_file, "a", encoding="utf-8") as f:
        for embedding in new_embeddings:
            f.write(json.dumps(embedding) + "\n")
            embeddings_file = None


# Function to check GPU temperature
def check_gpu_temperature():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    pynvml.nvmlShutdown()
    return temp


# Function to generate embeddings for vault content with checkpointing and logging
def generate_embeddings(vault_data, vault_name, start_idx=0):
    vault_embeddings = []
    progress_log = os.path.join(EMBEDDINGS_DIR, f"{vault_name}_progress.json")

    last_processed_chunk_id = None
    last_processed_file_path = None

    # Load progress log if exists
    if os.path.exists(progress_log):
        with open(progress_log, "r") as f:
            progress_data = json.load(f)
        last_processed_chunk_id = progress_data.get("last_processed_chunk_id")
        last_processed_file_path = progress_data.get("file_path")
        print(f"Last processed chunk ID: {last_processed_chunk_id}")
        print(f"File path: {last_processed_file_path}")
        # Optionally, you can use these values in your further processing
        # For example:
        # start_idx = progress_data.get("last_processed_idx", 0)
        # print(f"Resuming from index {start_idx}")

    else:
        print(f"Progress log '{progress_log}' does not exist.")

    for entry in tqdm(vault_data, desc="Generating embeddings"):
        file_path = entry["file_name"]
        modification_time = entry["modification_time"]
        chunks = entry["chunks"]
        """
        if last_processed_file_path and file_path != last_processed_file_path:
            continue  # Skip until the last processed file is found
        if last_processed_file_path and file_path == last_processed_file_path:
            last_processed_file_path = None  # Reset after resuming
        """
        for chunk in chunks:
            content = chunk["text"]
            chunk_id = chunk["id"]
            # print(chunk_id)
            # print(content, chunk_id)

            # Check GPU temperature before processing each content
            while (
                check_gpu_temperature() > 51
            ):  # Adjust temperature threshold as needed
                print(
                    RED
                    + "GPU temp is too high. Pausing Temperorily..."
                    + RESET_COLOR
                    + "\n"
                )
                time.sleep(30)  # Sleep for 30 seconds before rechecking
            """
            if last_processed_chunk_id and chunk_id != last_processed_chunk_id:
                continue  # Skip until the last processed chunk is found
            if last_processed_chunk_id and chunk_id == last_processed_chunk_id:
                last_processed_chunk_id = None  # Reset after resuming
            if last_processed_chunk_id and not (
                last_processed_chunk_id and last_processed_file_path
            ):
                continue  # Skip until both chunk and file are found
            """
            # response = ollama.embeddings(model="mxbai-embed-large", prompt=content)
            try:
                response = ollama.embeddings(model="mxbai-embed-large", prompt=content)

                if "embedding" in response:
                    vault_embeddings.append({chunk_id: response["embedding"]})
                else:
                    print(f"Failed to get embedding for content: {chunk_id}")

                # Save checkpoint every 10 embeddings or at the end of each entry
                if len(vault_embeddings) % 100 == 0 or entry == vault_data[-1]:
                    save_embeddings_txt(vault_embeddings)
                    vault_embeddings = []
                    data_to_save = {
                        "last_processed_chunk_id": chunk_id,
                        "file_path": file_path,
                    }
                    with open(progress_log, "w") as f:
                        json.dump(data_to_save, f)

            except Exception as e:
                print(f"Error processing chunk {chunk_id} in file {file_path}: {e}")

    return vault_embeddings


def load_embeddings_txt(embeddings_file):
    embeddings = []
    if os.path.exists(embeddings_file):
        with open(embeddings_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    data = json.loads(line.strip())
                    embeddings.append(list(data.keys())[0])  # Extract the ID
                except json.JSONDecodeError as e:
                    start = max(0, e.pos - 20)
                    end = min(len(line), e.pos + 20)
                    print(
                        f"JSONDecodeError: Extra data at line {line_number}, column {e.pos}:\n{line[start:end]}"
                    )
        return embeddings


def filter_vault_data(vault_data, embeddings_ids):
    filtered_vault_data = []
    for entry in vault_data:
        filtered_chunks = []
        for chunk in entry["chunks"]:
            if chunk["id"] not in embeddings_ids:
                filtered_chunks.append(chunk)
        if filtered_chunks:
            entry["chunks"] = filtered_chunks
            filtered_vault_data.append(entry)
    return filtered_vault_data


def clean_vault_json_file(vault_data):
    # Load IDs from vault_embeddings.txt
    embeddings_ids = load_embeddings_txt(TXT_EMBEDDINGS_FILE)
    if embeddings_ids:
        # Filter vault_data based on embeddings_ids
        filtered_vault_data = filter_vault_data(vault_data, embeddings_ids)

        # Write vault.json
        with open(UPDATED_VAULT_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered_vault_data, f, indent=2)

        print(
            f"Filtered vault.json based on embeddings and saved as {UPDATED_VAULT_JSON_FILE}"
        )
        return filtered_vault_data
    else:
        return vault_data


def main():
    # Locate vault.json automatically
    vault_data = read_vault_data("vault.json")
    print(f"Loaded vault.json ")
    # sending in the cleaned json data to the same vault_data
    vault_data = clean_vault_json_file(vault_data)
    print(f"cleaned vault.json ")
    vault_name = os.path.splitext(os.path.basename("vault.json"))[0]
    total_chunks = sum(len(file_dict["chunks"]) for file_dict in vault_data)

    saved_embeddings = load_embeddings(vault_name)
    vault_embeddings = None

    if os.path.exists(
        os.path.join(EMBEDDINGS_DIR, f"{vault_name}_embeddings.pt")
    ) and total_chunks == len(saved_embeddings):
        print(f"Loaded saved embeddings for {vault_name}")
        vault_embeddings = saved_embeddings
    else:
        print(f"Generating new embeddings for {vault_name}")
        # Create separate threads for generating and saving embeddings

        generate_thread = threading.Thread(
            target=generate_embeddings, args=(vault_data, vault_name)
        )
        save_thread = threading.Thread(
            target=save_embeddings_txt, args=(vault_embeddings,)
        )

        # Start the threads
        generate_thread.start()
        save_thread.start()

        # Wait for both threads to complete
        generate_thread.join()
        save_thread.join()

    if os.path.exists(TXT_EMBEDDINGS_FILE):
        convert_txt_to_pt(TXT_EMBEDDINGS_FILE, PT_EMBEDDINGS_FILE)
        print(f"Converted {TXT_EMBEDDINGS_FILE} to {PT_EMBEDDINGS_FILE}")
    else:
        print(f"{TXT_EMBEDDINGS_FILE} does not exist.")

    clean_vault_json_file(vault_data)

    print(f"Embeddings generation completed for {vault_name}")


if __name__ == "__main__":
    main()
