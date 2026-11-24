import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def open_local_collection(path, name):
    """Open an existing local corpus without an embedding function."""
    import chromadb
    from chromadb.config import Settings

    if not Path(path).is_dir():
        raise FileNotFoundError("The local Chroma directory is missing.")
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(name, embedding_function=None)


def search_documents(collection, query, limit=5):
    """Rank stored document text by lexical relevance without a model call."""
    terms = set(re.findall(r"\w+", query.casefold()))
    if not terms or limit <= 0:
        return []

    rows = collection.get(include=["documents", "metadatas"])
    documents = [
        (doc_id, document, metadata or {})
        for doc_id, document, metadata in zip(
            rows["ids"], rows["documents"], rows["metadatas"]
        )
        if document and document.strip()
    ]
    if not documents:
        return []

    tokenized = [re.findall(r"\w+", document.casefold()) for _, document, _ in documents]
    if not any(terms.intersection(tokens) for tokens in tokenized):
        return []
    scores = BM25Okapi(tokenized).get_scores(sorted(terms))
    matches = [
        {
            "id": doc_id,
            "document": document,
            "metadata": metadata,
            "relevance_score": float(score),
        }
        for (doc_id, document, metadata), tokens, score in zip(documents, tokenized, scores)
        if terms.intersection(tokens)
    ]
    return sorted(matches, key=lambda row: (-row["relevance_score"], row["id"]))[:limit]
