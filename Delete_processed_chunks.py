import os
import json

# Constants
EMBEDDINGS_DIR = "Embeddings"
VAULT_JSON_FILE = "vault.json"
VAULT_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "vault_embeddings.txt")
UPDATED_VAULT_JSON_FILE = "updated_vault.json"


def load_embeddings_txt(embeddings_file):
    embeddings = []
    with open(embeddings_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            embeddings.append(list(data.keys())[0])  # Extract the ID
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


def main():
    # Load IDs from vault_embeddings.txt
    embeddings_ids = load_embeddings_txt(VAULT_EMBEDDINGS_FILE)

    # Load vault.json
    with open(VAULT_JSON_FILE, "r", encoding="utf-8") as f:
        vault_data = json.load(f)

    # Filter vault_data based on embeddings_ids
    filtered_vault_data = filter_vault_data(vault_data, embeddings_ids)

    # Write updated_vault.json
    with open(UPDATED_VAULT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_vault_data, f, indent=2)

    print(
        f"Filtered vault.json based on embeddings and saved as {UPDATED_VAULT_JSON_FILE}"
    )


if __name__ == "__main__":
    main()
