from collections.abc import Mapping

from chromadb.errors import InternalError, NotFoundError, UniqueConstraintError


DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"


class EmbeddingContractError(ValueError):
    """The collection cannot be used with the requested embedding model."""


def _valid_model(model):
    return isinstance(model, str) and bool(model) and model == model.strip()


def _require_model(requested_model):
    if not _valid_model(requested_model):
        raise EmbeddingContractError("A valid embedding_model is required.")


def validate_embedding_compatibility(collection, requested_model, *, for_write=False):
    """Return False only for an unlabelled reader proven to be empty."""
    _require_model(requested_model)
    try:
        metadata = collection.metadata
    except Exception as error:
        raise EmbeddingContractError("Collection embedding_model metadata is unreadable.") from error
    if metadata is not None and not isinstance(metadata, Mapping):
        raise EmbeddingContractError("Collection embedding_model metadata is unreadable.")

    if metadata is not None and "embedding_model" in metadata:
        stored_model = metadata["embedding_model"]
        if not _valid_model(stored_model):
            raise EmbeddingContractError("Collection has invalid embedding_model metadata.")
        if stored_model != requested_model:
            raise EmbeddingContractError(
                f"Collection was indexed with {stored_model}, but the request uses {requested_model}."
            )
        return True

    if for_write:
        raise EmbeddingContractError(
            "Collection is missing embedding_model metadata. Select a labelled collection "
            "or use a new collection name."
        )

    try:
        count = collection.count()
    except Exception as error:
        raise EmbeddingContractError(
            "Collection is missing embedding_model metadata and its count is unavailable."
        ) from error
    if type(count) is not int or count < 0:
        raise EmbeddingContractError(
            "Collection is missing embedding_model metadata and its count is invalid."
        )
    if count == 0:
        return False
    raise EmbeddingContractError(
        "Non-empty collection is missing embedding_model metadata. Select a labelled collection."
    )


def open_collection_for_embedding_write(client, name, model):
    """Require an existing label or label a genuinely new collection at creation."""
    _require_model(model)
    try:
        collection = client.get_collection(name=name, embedding_function=None)
    except NotFoundError:
        try:
            collection = client.create_collection(
                name=name, metadata={"embedding_model": model}, embedding_function=None
            )
        except UniqueConstraintError:
            collection = client.get_collection(name=name, embedding_function=None)
        except InternalError as error:
            # Chroma's Rust client uses InternalError for this duplicate-create case.
            if str(error) != f"Collection [{name}] already exists":
                raise
            collection = client.get_collection(name=name, embedding_function=None)

    validate_embedding_compatibility(collection, model, for_write=True)
    return collection


def ensure_collection_embedding_model(collection, model_name):
    """Validate a writable collection while leaving its metadata intact."""
    validate_embedding_compatibility(collection, model_name, for_write=True)
    return model_name


def assert_embedding_model_matches(collection, requested_model):
    validate_embedding_compatibility(collection, requested_model)
