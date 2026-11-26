import re
from pathlib import Path


def open_local_collection(path, name):
    """Open an existing local corpus without an embedding function."""
    import chromadb
    from chromadb.config import Settings

    if not (Path(path) / "chroma.sqlite3").is_file():
        raise FileNotFoundError("The local Chroma database is missing.")
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(name, embedding_function=None)


def search_documents(collection, query, limit=5, *, where=None):
    """Rank stored document text by lexical relevance without a model call."""
    if limit <= 0 or not re.search(r"[^\W_]", query):
        return []

    from chroma_hybrid_retrieval import search_collection

    rows = search_collection(collection, query, top_k=limit, where=where)
    return [
        {
            "id": row["chunk_id"],
            "chunk_id": row["chunk_id"],
            "collection_id": row["collection_id"],
            "document": row["document"],
            "metadata": row["meta"],
            "relevance_score": row["bm25_score"],
        }
        for row in rows
    ]
