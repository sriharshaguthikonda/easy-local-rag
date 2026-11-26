import importlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
    assert store.query(query_embeddings=[[1.0, 0.0]], n_results=1)["ids"] == [["first"]]
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
from pathlib import Path
from time import monotonic, sleep
from chroma_hybrid_retrieval import store_lock
client = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False))
collection = client.get_collection("synthetic_hybrid", embedding_function=None)
with store_lock(sys.argv[1]):
    collection.update(ids=["first"], documents=["first new"], embeddings=[[0.0, 1.0]])
    print("first update held", flush=True)
    deadline = monotonic() + 20
    while not Path(sys.argv[2]).exists():
        if monotonic() >= deadline:
            raise RuntimeError("test reader did not release the writer")
        sleep(0.025)
    collection.update(
        ids=["second"], documents=["second new"], embeddings=[[1.0, 0.0]],
        metadatas=[{"revision": 2}],
    )
"""
    release = Path(path) / "release-writer"
    with subprocess.Popen(
        [sys.executable, "-c", writer, path, str(release)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as process, ThreadPoolExecutor(2) as pool:
        try:
            assert pool.submit(process.stdout.readline).result(timeout=15).strip() == "first update held"
            search = pool.submit(vector_search)
            with pytest.raises(TimeoutError):
                search.result(timeout=1)
        finally:
            release.touch()
        assert process.wait(timeout=15) == 0, process.stderr.read()
        context, sources = search.result(timeout=30)
    assert store.count() == 2
    assert context == "second new"
    assert sources[0]["chunk_id"] == "second"
    assert sources[0]["revision"] == 2


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


def test_hybrid_legs_and_metadata_filters_use_the_same_eligible_scope(store):
    from chroma_hybrid_retrieval import search_collection

    near_ids = [f"near-{index:03}" for index in range(55)]
    _add(store, near_ids + ["distant"],
         ["ordinary passage"] * 55 + ["uniquelexeme passage"],
         [[1.0, 0.0]] * 55 + [[0.0, 1.0]])
    store.update(ids=["distant"], metadatas=[{"category": "include"}])
    kwargs = dict(
        embedding_model=DEFAULT_EMBEDDING_MODEL, embed=lambda: [1.0, 0.0],
        vector_weight=0.4, lexical_weight=0.6, top_k=2,
    )
    hits = search_collection(store, "uniquelexeme", **kwargs)
    assert hits[0]["chunk_id"] == "distant"
    assert hits[0]["distance"] is None
    assert hits[1]["chunk_id"] in near_ids and hits[1]["distance"] == 0
    assert {row["collection_id"] for row in hits} == {str(store.id)}
    hits = search_collection(store, "uniquelexeme", where={"category": "include"}, **kwargs)
    assert [row["chunk_id"] for row in hits] == ["distant"]
    assert hits[0]["distance"] > 0 and hits[0]["bm25_score"] > 0
    store.update(ids=["distant"], metadatas=[{"category": "exclude"}])
    assert search_collection(store, "uniquelexeme", where={"category": "include"}, **kwargs) == []


def test_collection_switch_and_same_name_recreation(store):
    from chroma_hybrid_retrieval import RetrievalError, search_collection, search_store

    _add(store, ["same-id"], ["uniquelexeme original"], [[1.0, 0.0]])
    client = chromadb.PersistentClient(
        path=store._client.get_settings().persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )
    second = client.create_collection("synthetic_switch", embedding_function=None)
    _add(second, ["same-id"], ["uniquelexeme second"], [[1.0, 0.0]])
    assert search_collection(store, "uniquelexeme")[0]["collection_id"] == str(store.id)
    switched = search_collection(second, "uniquelexeme")[0]
    assert switched["document"] == "uniquelexeme second"
    assert switched["collection_id"] == str(second.id)
    client.delete_collection(store.name)
    replacement = client.create_collection(store.name, embedding_function=None)
    _add(replacement, ["same-id"], ["uniquelexeme replacement"], [[1.0, 0.0]])
    with pytest.raises(RetrievalError, match="collection changed"):
        search_collection(store, "uniquelexeme")
    current = search_store(client.get_settings().persist_directory, replacement.name, "uniquelexeme")
    assert current[0]["document"] == "uniquelexeme replacement"
    assert current[0]["collection_id"] == str(replacement.id)


def test_capture_limits_leave_a_clean_lexical_index(store, monkeypatch):
    import chroma_hybrid_retrieval as hybrid

    _add(store, ["small", "large"], ["needle", "needle " * 20], [[1.0, 0.0]] * 2)
    with monkeypatch.context() as limited:
        limited.setattr(hybrid, "MAX_CHUNK_BYTES", 100)
        with pytest.raises(hybrid.RetrievalError, match="text limit"):
            hybrid.capture(store, "needle")
        limited.setattr(hybrid, "MAX_CHUNK_BYTES", 512 * 1024)
        limited.setattr(hybrid, "MAX_EVIDENCE_BYTES", 80)
        with pytest.raises(hybrid.RetrievalError, match="evidence size limit"):
            hybrid.capture(store, "needle")
    assert {row["chunk_id"] for row in hybrid.capture(store, "needle")} == {"small", "large"}


def test_large_real_collection_lexical_index_is_paged(store, monkeypatch):
    from contextlib import contextmanager
    from time import perf_counter
    import tracemalloc

    import chroma_hybrid_retrieval as hybrid

    count = 25_601
    target = f"synthetic-{count - 1:05}"
    for offset in range(0, count, 512):
        keys = [f"synthetic-{i:05}" for i in range(offset, min(offset + 512, count))]
        docs = ["generic passage " * 25 + ("rarelexeme" if key == target else key) for key in keys]
        _add(store, keys, docs, [[1.0, 0.0]] * len(keys))

    pages = []
    sqlite_stats = {}
    original_pages = hybrid._pages
    original_index = hybrid._lexical_index

    def tracked_pages(*args):
        for page in original_pages(*args):
            rows = list(page)
            pages.append(len(rows))
            yield iter(rows)

    @contextmanager
    def tracked_index(enabled):
        with original_index(enabled) as index:
            yield index
            sqlite_stats["bytes"] = index.execute("PRAGMA page_count").fetchone()[0] * index.execute(
                "PRAGMA page_size"
            ).fetchone()[0]
            sqlite_stats["cache_kib"] = -index.execute("PRAGMA cache_size").fetchone()[0]

    monkeypatch.setattr(hybrid, "_pages", tracked_pages)
    monkeypatch.setattr(hybrid, "_lexical_index", tracked_index)
    tracemalloc.start()
    started = perf_counter()
    try:
        hits = hybrid.capture(store, "rarelexeme", top_k=1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = perf_counter() - started
    assert [row["chunk_id"] for row in hits] == [target]
    assert sum(pages) == count and max(pages) <= 512
    assert peak < 24 * 1024**2
    assert sqlite_stats["cache_kib"] == 4096
    print(
        f"retrieval benchmark: {count} rows; {len(pages)} pages; max {max(pages)}; "
        f"{elapsed:.2f}s; Python peak {peak / 1024**2:.2f} MiB; "
        f"SQLite scratch {sqlite_stats['bytes'] / 1024**2:.2f} MiB; "
        f"SQLite cache {sqlite_stats['cache_kib']} KiB"
    )
