import torch
import json
import os

# Constants
EMBEDDINGS_DIR = "Embeddings"
PT_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "vault_embeddings.pt")
TXT_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "vault_embeddings.txt")


def convert_pt_to_txt(pt_file, txt_file):
    embeddings = torch.load(pt_file)
    with open(txt_file, "w", encoding="utf-8") as f:
        for embedding in embeddings:
            f.write(json.dumps(embedding) + "\n")


if os.path.exists(PT_EMBEDDINGS_FILE):
    convert_pt_to_txt(PT_EMBEDDINGS_FILE, TXT_EMBEDDINGS_FILE)
    print(f"Converted {PT_EMBEDDINGS_FILE} to {TXT_EMBEDDINGS_FILE}")
else:
    print(f"{PT_EMBEDDINGS_FILE} does not exist.")
