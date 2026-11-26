"""Helper script to run ChromaDB queries in isolation from Qt.
Called via subprocess to avoid native library conflicts.
"""
import sys
import json

import chromadb
import ollama
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

from embedding_contract import validate_embedding_compatibility


def query_chromadb(chromadb_path, collection_name, query, embedding_model, n_results=20):
    client = chromadb.PersistentClient(
        path=chromadb_path,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = client.get_collection(collection_name, embedding_function=None)
    if not validate_embedding_compatibility(collection, embedding_model):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    embedding = ollama.embeddings(model=embedding_model, prompt=query, keep_alive=-1)["embedding"]

    result = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return result


def main():
    data = json.loads(sys.stdin.read())

    try:
        result = query_chromadb(
            data["chromadb_path"], data["collection_name"], data["query"],
            data["embedding_model"], data.get("n_results", 20),
        )

        output = {
            "success": True,
            "documents": result.get("documents", [[]])[0],
            "metadatas": result.get("metadatas", [[]])[0],
            "distances": result.get("distances", [[]])[0],
        }
    except Exception as e:
        output = {"success": False, "error": str(e)}

    print(json.dumps(output))


if __name__ == "__main__":
    main()
