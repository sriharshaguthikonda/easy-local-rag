from unittest.mock import Mock

import pytest
from chromadb.errors import InternalError, NotFoundError, UniqueConstraintError

import embedding_contract as contract


MODEL = "mxbai-embed-large"


def collection(metadata, count=0):
    result = Mock(metadata=metadata)
    if isinstance(count, Exception):
        result.count.side_effect = count
    else:
        result.count.return_value = count
    return result


@pytest.mark.parametrize("count", [RuntimeError("backend unavailable"), False, "0", 0.0, -1])
@pytest.mark.parametrize("for_write", [False, True])
def test_unknown_count_error_rejects_read_and_write(count, for_write):
    target = collection({"unrelated": "keep"}, count)
    guard = (
        contract.ensure_collection_embedding_model if for_write
        else contract.assert_embedding_model_matches
    )
    with pytest.raises(ValueError):
        guard(target, MODEL)
    target.modify.assert_not_called()
    target.add.assert_not_called()


@pytest.mark.parametrize("count", [0, 5])
def test_existing_collection_metadata_is_never_modified(count):
    target = collection({"unrelated": "keep"}, count)
    with pytest.raises(ValueError, match="embedding_model"):
        contract.ensure_collection_embedding_model(target, MODEL)
    assert target.metadata == {"unrelated": "keep"}
    target.modify.assert_not_called()


@pytest.mark.parametrize("label", ["", 4, False])
def test_malformed_stored_model_is_never_verified_empty(label):
    target = collection({"embedding_model": label})
    with pytest.raises(ValueError):
        contract.assert_embedding_model_matches(target, MODEL)
    target.modify.assert_not_called()


def test_named_match_needs_no_count_and_preserves_metadata():
    target = collection({"embedding_model": MODEL, "unrelated": "keep"}, RuntimeError())
    assert contract.ensure_collection_embedding_model(target, MODEL) == MODEL
    contract.assert_embedding_model_matches(target, MODEL)
    target.count.assert_not_called()
    target.modify.assert_not_called()


def test_assert_mismatch_raises():
    target = collection({"embedding_model": "other-model"}, 5)
    with pytest.raises(ValueError, match="other-model"):
        contract.assert_embedding_model_matches(target, MODEL)
    target.modify.assert_not_called()


@pytest.mark.parametrize("conflict", [
    UniqueConstraintError("duplicate"),
    InternalError("Collection [synthetic_test] already exists"),
])
@pytest.mark.parametrize("label", [MODEL, "other-model"])
def test_writer_create_race_revalidates_collection(conflict, label):
    winner = collection({"embedding_model": label, "unrelated": "keep"})
    client = Mock()
    client.get_collection.side_effect = [NotFoundError("absent"), winner]
    client.create_collection.side_effect = conflict
    # The baseline uses get_or_create and then tags empty collections.
    client.get_or_create_collection.return_value = winner

    def open_writer():
        opener = getattr(contract, "open_collection_for_embedding_write", None)
        if opener:
            return opener(client, "synthetic_test", MODEL)
        target = client.get_or_create_collection(name="synthetic_test")
        contract.ensure_collection_embedding_model(target, MODEL)
        return target

    if label == MODEL:
        assert open_writer() is winner
    else:
        with pytest.raises(ValueError, match="other-model"):
            open_writer()
    client.create_collection.assert_called_once_with(
        name="synthetic_test", metadata={"embedding_model": MODEL}, embedding_function=None
    )
    assert client.get_collection.call_count == 2
    winner.modify.assert_not_called()
