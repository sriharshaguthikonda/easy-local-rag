import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embedding_contract import EmbeddingContractError
from hybrid_rank_fusion import fuse_candidates


def test_native_chroma_retrieval_cases():
    # Keep Chroma's Rust writer outside the process that imports Qt and SDK fakes.
    suite = Path(__file__).with_name("chroma_hybrid_native_cases.py")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), "-q", "-s", "--tb=short", "-p", "no:cacheprovider"],
        cwd=suite.parents[1], capture_output=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


def test_fusion_is_deterministic_and_preserves_both_id_components():
    vector = {
        "collection_id": "a", "chunk_id": "same", "distance": 0.2,
        "embedding": [1.0, 0.0], "document": "same filename one",
    }
    shared = {"collection_id": "a", "chunk_id": "same", "bm25_score": 2.0, "distance": None}
    lexical = {"collection_id": "b", "chunk_id": "same", "distance": None, "bm25_score": 3.0}
    vector_only = {"collection_id": "a", "chunk_id": "tail", "distance": 0.3}
    rows = fuse_candidates(
        [vector, vector, vector_only], [lexical, shared, shared],
        vector_weight=1, lexical_weight=1,
    )
    assert [(item["collection_id"], item["chunk_id"]) for item in rows] == [
        ("a", "same"), ("b", "same"), ("a", "tail"),
    ]
    assert rows[0]["distance"] == 0.2 and rows[0]["embedding"] == [1.0, 0.0]
    assert rows[0]["bm25_score"] == 2.0
    assert [row["final_score"] for row in rows] == [1 / 61 + 1 / 62, 1 / 61, 1 / 62]
    assert "final_score" not in vector
    assert fuse_candidates(
        [vector], [lexical], vector_weight=1, lexical_weight=0,
    ) == [dict(vector, final_score=1 / 61)]
    assert fuse_candidates(
        [lexical], [vector], vector_weight=1, lexical_weight=1,
    )[0]["collection_id"] == "a"


@pytest.mark.parametrize("change", ["uuid", "model"])
def test_change_between_preflight_and_capture_blocks_vector_query(monkeypatch, tmp_path, change):
    import chroma_hybrid_retrieval as hybrid

    first = SimpleNamespace(id="original", metadata={"embedding_model": "fixture"}, count=lambda: 1)
    second = SimpleNamespace(
        id="replacement" if change == "uuid" else first.id,
        metadata={"embedding_model": "other"} if change == "model" else first.metadata,
        query=Mock(), count=lambda: 1,
    )
    opener = Mock(side_effect=[first, second])
    monkeypatch.setattr(hybrid, "_open_selection", opener)
    monkeypatch.setattr(hybrid, "store_lock", lambda _path: nullcontext())
    embed = Mock(return_value=[1, 0])
    expected = hybrid.RetrievalError if change == "uuid" else EmbeddingContractError
    with pytest.raises(expected, match="collection changed|indexed with"):
        hybrid.search_path_in_worker(
            tmp_path, "synthetic", "needle", embedding_model="fixture",
            embed=embed, vector_weight=1, lexical_weight=0,
        )
    embed.assert_called_once_with()
    second.query.assert_not_called()


def test_duplicate_ids_fail_instead_of_returning_partial_lexical_results():
    import chroma_hybrid_retrieval as hybrid

    target = SimpleNamespace(
        id="synthetic",
        get=lambda **_: {
            "ids": ["same", "same"], "documents": ["needle", "needle"], "metadatas": [{}, {}],
        },
    )
    with pytest.raises(hybrid.RetrievalError, match="UNIQUE constraint failed"):
        hybrid.capture(target, "needle")


@pytest.mark.parametrize("space, expected", [
    ("l2", {"a": 1.0, "b": 5.0}),
    ("cosine", {"a": 0.0, "b": 1.0}),
    ("ip", {"a": -1.0, "b": 1.0}),
])
def test_filtered_distances_follow_collection_metric(space, expected):
    import chroma_hybrid_retrieval as hybrid

    target = SimpleNamespace(
        configuration={"hnsw": {"space": space}}, metadata={},
        get=lambda **_: {"ids": ["b", "a"], "embeddings": [[0, 2], [2, 0]]},
    )
    assert hybrid._filtered_vector_hits(
        target, [1, 0], ["a", "b"], time.monotonic() + 5,
    ) == pytest.approx(expected)


def test_store_lock_releases_and_distinct_paths_do_not_contend(tmp_path, monkeypatch):
    import chroma_hybrid_retrieval as hybrid

    monkeypatch.setattr(hybrid, "LOCK_SECONDS", 0.1)
    with hybrid.store_lock(tmp_path):
        with pytest.raises(hybrid.RetrievalError, match="Timed out"):
            with hybrid.store_lock(tmp_path / "child" / ".."):
                pytest.fail("the same store path was acquired twice")
        with hybrid.store_lock(tmp_path / "other"):
            pass
    with hybrid.store_lock(tmp_path):
        pass
