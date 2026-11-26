import importlib
import json
from pathlib import Path
import subprocess
import sys

import chromadb
import ollama
import pytest
from chromadb.config import Settings

from embedding_contract import DEFAULT_EMBEDDING_MODEL
from rag_prompting import build_numbered_sources


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "FALSE")
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=Settings(anonymized_telemetry=False)
    )
    target = client.create_collection(
        "synthetic_hybrid", embedding_function=None,
        metadata={"embedding_model": DEFAULT_EMBEDDING_MODEL},
    )
    monkeypatch.setattr(ollama, "embeddings", lambda **_: {"embedding": [1.0, 0.0]})
    import nest_asyncio
    monkeypatch.setattr(nest_asyncio, "apply", lambda: None)
    return target


def _add(target, ids, documents, vectors, names=None):
    target.add(
        ids=ids, documents=documents, embeddings=vectors,
        metadatas=[
            {"file_name": name, "text": text, "category": "fixture"}
            for name, text in zip(names or [f"{key}.txt" for key in ids], documents)
        ],
    )


def _direct(target, monkeypatch, capsys, **options):
    import GUI_direct_search as direct

    args = dict(
        module_path=str(Path(direct.__file__).parent),
        path=target._client.get_settings().persist_directory,
        col_name=target.name, embedding_model=DEFAULT_EMBEDDING_MODEL,
        query="uniquelexeme", rewritten_input="uniquelexeme", synonym_dict={},
        top_k=2, alpha=1, beta=0, gamma=0, delta=0, mode="hybrid",
        include=["documents", "metadatas", "distances"],
    )
    monkeypatch.setenv("_DS_ARGS", json.dumps({**args, **options}))
    capsys.readouterr()
    exec(compile(direct._SEARCH_SCRIPT, "direct-search-subprocess", "exec"), {})
    return json.loads(capsys.readouterr().out)["results"]


def _backend(monkeypatch, target, name):
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "model", DEFAULT_EMBEDDING_MODEL)
    monkeypatch.setattr(module, "collection", target, raising=False)
    monkeypatch.setattr(module, "print_relevant_context", lambda _results: None)
    return module


def test_same_basename_preserves_both_chunks(store, monkeypatch, capsys):
    _add(store, ["first", "second"], ["first text", "second text"],
         [[1.0, 0.0], [0.5, 0.5]], ["same.txt", "same.txt"])
    rows = _direct(store, monkeypatch, capsys)
    assert [row["document"] for row in rows] == ["first text", "second text"]
    assert [(row["collection_id"], row["chunk_id"]) for row in rows] == [
        (str(store.id), "first"), (str(store.id), "second"),
    ]
    citations = build_numbered_sources(rows)
    assert [(citation["citation_id"], citation["chunk_id"]) for citation in citations] == [
        (1, "first"), (2, "second"),
    ]


def test_gui_view_filter_applies_before_top_k(store, monkeypatch, capsys):
    _add(store, ["near", "eligible"], ["near text", "other text"],
         [[1.0, 0.0], [0.0, 1.0]], ["near.txt", "Selected.TXT"])
    rows = _direct(store, monkeypatch, capsys, filter_text="sELECTed", top_k=1)
    assert [row["document"] for row in rows] == ["other text"]


@pytest.mark.parametrize("name", [
    "streamlit_groq_lama_chromadb_RAG_ETTS",
    "groq_lama_chromadb_RAG_ETTS",
    "context_retrieval",
])
def test_backend_recovers_lexical_only_hit(store, monkeypatch, name):
    vector_ids = [f"vector-{index:03}" for index in range(51)]
    _add(store, vector_ids + ["lexical-target"],
         ["common decoy passage"] * 51 + ["uniquelexeme evidence"],
         [[1.0, 0.0]] * 51 + [[0.0, 1.0]])
    assert "lexical-target" not in store.query(
        query_embeddings=[[1.0, 0.0]], n_results=50
    )["ids"][0]

    module = _backend(monkeypatch, store, name)
    weights = {"alpha": 0, "beta": 1} if name.startswith("streamlit") else {
        "alpha": 0, "beta": 0, "gamma": 1
    }
    context = module.get_relevant_context_hybrid(
        "uniquelexeme", top_k=1, additional_unique_files=0, rewrite=False, **weights
    )
    if name.startswith("streamlit"):
        context, sources = context
        assert [source["document"] for source in sources] == ["uniquelexeme evidence"]
        assert sources[0]["chunk_id"] == "lexical-target"
        assert sources[0]["collection_id"] == str(store.id)
    assert context == "uniquelexeme evidence"


def test_same_count_ingestion_refreshes_native_vector_candidates(store, monkeypatch):
    _add(store, ["first", "second"], ["first old", "second old"],
         [[1.0, 0.0], [0.0, 1.0]])
    backend = _backend(monkeypatch, store, "streamlit_groq_lama_chromadb_RAG_ETTS")
    def vector_search():
        return backend.get_relevant_context_hybrid(
            "query", top_k=1, additional_unique_files=0,
            keyword_match=False, alpha=1, beta=0, rewrite=False,
        )
    assert vector_search()[0] == "first old"
    path = store._client.get_settings().persist_directory
    writer = """
import sys
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False))
collection = client.get_collection("synthetic_hybrid", embedding_function=None)
collection.update(ids=["second"], documents=["second new"], embeddings=[[1.0, 0.0]])
collection.update(ids=["first"], documents=["first new"], embeddings=[[0.0, 1.0]])
"""
    subprocess.run([sys.executable, "-c", writer, path], check=True, timeout=30)
    assert store.count() == 2
    context, sources = vector_search()
    assert context == "second new"
    assert sources[0]["chunk_id"] == "second"


def test_lexical_read_tracks_same_count_edits_delete_and_insert(store):
    from offline_search import search_documents

    _add(store, ["one"], ["oldlexeme survives"], [[1.0, 0.0]])
    assert [row["id"] for row in search_documents(store, "oldlexeme")] == ["one"]
    store.update(ids=["one"], documents=["newlexeme survives"], embeddings=[[1.0, 0.0]])
    assert store.count() == 1
    assert search_documents(store, "oldlexeme") == []
    assert [row["id"] for row in search_documents(store, "newlexeme")] == ["one"]
    store.delete(ids=["one"])
    _add(store, ["two"], ["newlexeme reappears"], [[1.0, 0.0]])
    assert [row["id"] for row in search_documents(store, "newlexeme")] == ["two"]
