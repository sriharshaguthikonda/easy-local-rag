"""Helper script to run ChromaDB queries in isolation from Qt.
Called via subprocess to avoid native library conflicts.
"""
import sys
import json

import ollama

from chroma_hybrid_retrieval import search_keywords, search_path_in_worker, self_test_query


def query_chromadb(chromadb_path, collection_name, query, embedding_model, n_results=20):
    result = self_test_query(
        chromadb_path, collection_name, embedding_model, None,
        lambda: ollama.embeddings(model=embedding_model, prompt=query, keep_alive=-1)["embedding"],
        n_results, ["documents", "metadatas", "distances"],
    )
    return result


def _hybrid_query(data):
    model = data["embedding_model"]
    rewritten = data["rewritten"]
    words = search_keywords(rewritten, data["synonyms"])
    rows = search_path_in_worker(
        data["chromadb_path"], data["collection_name"], data["query"],
        embedding_model=model, expected_id=data.get("collection_id"),
        embed=lambda: ollama.embeddings(model=model, prompt=rewritten, keep_alive=-1)["embedding"],
        top_k=data["n_results"], vector_weight=data["vector_weight"],
        lexical_weight=data["lexical_weight"], keyword_weight=data["keyword_weight"],
        words=words,
    )
    for row in rows:
        row["keywords"] = words
    return rows


def main():
    data = json.load(sys.stdin)
    try:
        if data.get("hybrid"):
            output = {"success": True, "results": _hybrid_query(data)}
        else:
            result = query_chromadb(
                data["chromadb_path"], data["collection_name"], data["query"],
                data["embedding_model"], data.get("n_results", 20),
            )
            output = {
                "success": True,
                "collection_id": result.get("collection_id"),
                "ids": result["ids"][0],
                "documents": result["documents"][0],
                "metadatas": result["metadatas"][0],
                "distances": result["distances"][0],
            }
    except Exception as error:
        output = {"success": False, "error": str(error)}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
