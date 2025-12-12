import sys
from pathlib import Path

import pytest
from PyQt5.QtCore import QCoreApplication

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from GUI_workers import ChromaDBSearchWorker  # noqa: E402


# Guarantee a Qt application exists for signal delivery during tests
app = QCoreApplication.instance() or QCoreApplication([])


class DummyCollection:
    def __init__(self, name="dummy"):
        self.name = name
        self.last_query_kwargs = None
        self.query_result = None
        self.raise_error = None

    def query(self, **kwargs):
        self.last_query_kwargs = kwargs
        if self.raise_error:
            raise self.raise_error
        return self.query_result


def test_chromadb_search_worker_success(monkeypatch):
    """Worker should emit formatted results when query succeeds."""
    # Mock embedding output
    mock_embedding = [0.1, 0.2, 0.3]

    def fake_embeddings(model, prompt, keep_alive):
        return {"embedding": mock_embedding}

    monkeypatch.setattr("GUI_workers.ollama.embeddings", fake_embeddings)

    # Prepare fake collection result
    collection = DummyCollection(name="test_collection")
    collection.query_result = {
        "metadatas": [[{"file_name": "a.txt"}, {"file_name": "b.txt"}]],
        "documents": [["doc A", "doc B"]],
        "distances": [[0.1, 0.2]],
    }

    worker = ChromaDBSearchWorker("hello", collection, embedding_model="test-embed", n_results=2)

    results = []
    worker.results_ready.connect(results.append)

    # Run synchronously (no thread start) for testability
    worker.run()

    assert len(results) == 1
    formatted = results[0]
    assert len(formatted) == 2
    assert formatted[0]["file_name"] == "a.txt"
    assert pytest.approx(formatted[0]["similarity"], rel=1e-6) == 0.9
    assert collection.last_query_kwargs["query_embeddings"] == [mock_embedding]
    assert collection.last_query_kwargs["n_results"] == 2


def test_chromadb_search_worker_query_error(monkeypatch):
    """Worker should emit error_occurred when collection.query raises."""
    def fake_embeddings(model, prompt, keep_alive):
        return {"embedding": [0.1, 0.2]}

    monkeypatch.setattr("GUI_workers.ollama.embeddings", fake_embeddings)

    collection = DummyCollection(name="test_collection")
    collection.raise_error = RuntimeError("boom")

    worker = ChromaDBSearchWorker("hello", collection, embedding_model="test-embed", n_results=2)

    errors = []
    worker.error_occurred.connect(errors.append)

    worker.run()

    assert errors, "Expected an error emission"
    assert "boom" in errors[0]
