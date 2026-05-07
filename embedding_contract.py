from typing import Any


DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"


def get_collection_embedding_model(collection: Any) -> str | None:
    metadata = getattr(collection, "metadata", None) or {}
    return metadata.get("embedding_model")


def _collection_count(collection: Any) -> int:
    try:
        return int(collection.count())
    except Exception:
        return 0


def ensure_collection_embedding_model(collection: Any, model_name: str) -> str:
    current = get_collection_embedding_model(collection)
    if current:
        if current != model_name:
            raise ValueError(
                f"Collection embedding model mismatch: stored={current}, requested={model_name}"
            )
        return current

    count = _collection_count(collection)
    if count > 0:
        raise ValueError(
            "Collection is missing embedding_model metadata and is non-empty. "
            "Manual migration is required."
        )

    metadata = getattr(collection, "metadata", None) or {}
    updated = {**metadata, "embedding_model": model_name}
    if hasattr(collection, "modify"):
        collection.modify(metadata=updated)
    return model_name


def assert_embedding_model_matches(collection: Any, requested_model: str) -> None:
    current = get_collection_embedding_model(collection)
    if not current:
        count = _collection_count(collection)
        if count > 0:
            raise ValueError(
                "Collection missing embedding_model metadata; cannot verify query model."
            )
        return
    if current != requested_model:
        raise ValueError(
            f"Collection was indexed with {current}, but query uses {requested_model}."
        )
