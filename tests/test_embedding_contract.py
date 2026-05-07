from embedding_contract import (
    assert_embedding_model_matches,
    ensure_collection_embedding_model,
)


class DummyCollection:
    def __init__(self, metadata=None, count_value=0):
        self.metadata = metadata
        self._count = count_value
        self.modified_to = None

    def count(self):
        return self._count

    def modify(self, metadata=None):
        self.metadata = metadata or {}
        self.modified_to = self.metadata


def test_ensure_sets_metadata_for_empty_collection():
    collection = DummyCollection(metadata={}, count_value=0)
    ensure_collection_embedding_model(collection, "mxbai-embed-large")
    assert collection.metadata["embedding_model"] == "mxbai-embed-large"


def test_ensure_rejects_missing_metadata_on_non_empty_collection():
    collection = DummyCollection(metadata={}, count_value=10)
    try:
        ensure_collection_embedding_model(collection, "mxbai-embed-large")
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "Manual migration is required" in str(error)


def test_assert_mismatch_raises():
    collection = DummyCollection(metadata={"embedding_model": "a"}, count_value=5)
    try:
        assert_embedding_model_matches(collection, "b")
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "indexed with a" in str(error)
