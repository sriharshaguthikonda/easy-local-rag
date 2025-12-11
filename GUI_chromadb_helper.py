"""Helper script to run ChromaDB queries in isolation from Qt.
Called via subprocess to avoid native library conflicts.
"""
import sys
import json

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


def query_chromadb(chromadb_path, collection_name, embedding, n_results=20):
    client = chromadb.PersistentClient(
        path=chromadb_path,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = client.get_collection(collection_name)

    result = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return result


def main():
    data = json.loads(sys.stdin.read())

    chromadb_path = data["chromadb_path"]
    collection_name = data["collection_name"]
    embedding = data["embedding"]
    n_results = data.get("n_results", 20)

    try:
        result = query_chromadb(chromadb_path, collection_name, embedding, n_results)

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
