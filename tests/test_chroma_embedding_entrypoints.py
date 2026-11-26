import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import chromadb
import ollama
import pytest
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from embedding_contract import DEFAULT_EMBEDDING_MODEL as MODEL
from offline_search import search_documents


class VectorQueryReached(BaseException):
    """Stop at a synthetic vector boundary before ranking or other work."""


def synthetic_collection(metadata=None, count=1):
    target = Mock(metadata=metadata)
    target.name = "synthetic_test"
    target.count.return_value = count
    target.query.side_effect = VectorQueryReached
    return target


def stub_embeddings(monkeypatch, dimensions=2):
    embed = Mock(return_value={"embedding": [1.0] + [0.0] * (dimensions - 1)})
    monkeypatch.setattr(ollama, "embeddings", embed)
    return embed


@pytest.fixture(autouse=True)
def isolate_async_loop_setup(monkeypatch):
    import nest_asyncio

    # Windows asyncio's startup creates a loopback socket pair. The retrieval
    # tests run synchronously and neither need nor exercise that startup hook.
    monkeypatch.setattr(nest_asyncio, "apply", lambda: None)


@pytest.fixture
def temp_chroma(tmp_path):
    return chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False)
    )


@pytest.mark.parametrize("module_name", [
    "streamlit_groq_lama_chromadb_RAG_ETTS", "groq_lama_chromadb_RAG_ETTS",
    "context_retrieval",
])
@pytest.mark.parametrize("state", ["unlabelled", "unreadable", "same-dimension", "different-dimension", "match", "empty"])
def test_maintained_backend_checks_before_embedding(monkeypatch, module_name, state):
    backend = importlib.import_module(module_name)
    metadata = {"embedding_model": MODEL if state == "match" else "other-model"}
    if state in {"unlabelled", "unreadable", "empty"}:
        metadata = {}
    target = synthetic_collection(metadata, count=0 if state == "empty" else 1)
    if state == "unreadable":
        target.count.side_effect = RuntimeError("count unavailable")
    embed = stub_embeddings(monkeypatch, dimensions=3 if state == "different-dimension" else 2)
    monkeypatch.setattr(backend, "model", MODEL)
    monkeypatch.setattr(backend, "collection", target)

    def retrieve():
        return backend.get_relevant_context_hybrid("needle", rewrite=False)

    if state == "match":
        with pytest.raises(VectorQueryReached):
            retrieve()
        embed.assert_called_once_with(model=MODEL, prompt="needle", keep_alive=-1)
        target.query.assert_called_once()
    elif state == "empty":
        assert retrieve() == (("", []) if module_name.startswith("streamlit") else "")
        embed.assert_not_called()
        target.query.assert_not_called()
    else:
        with pytest.raises(RuntimeError) as failure:
            retrieve()
        assert "embedding_model" in str(failure.value.__cause__) or "indexed with" in str(failure.value.__cause__)
        embed.assert_not_called()
        target.query.assert_not_called()


def test_real_chroma_refuses_different_model_with_equal_dimensions(monkeypatch, temp_chroma):
    import streamlit_groq_lama_chromadb_RAG_ETTS as backend

    target = temp_chroma.create_collection(
        "synthetic_vectors", embedding_function=None, metadata={"embedding_model": "other-model"}
    )
    target.add(ids=["test-id"], documents=["needle"], embeddings=[[1.0, 0.0]],
               metadatas=[{"file_name": "fixture.txt"}])
    embed = stub_embeddings(monkeypatch)
    monkeypatch.setattr(backend, "collection", target)
    monkeypatch.setattr(backend, "model", MODEL)
    with pytest.raises(RuntimeError):
        backend.get_relevant_context_hybrid(
            "needle", top_k=1, additional_unique_files=0, keyword_match=False, rewrite=False
        )
    embed.assert_not_called()
    assert search_documents(target, "needle")[0]["id"] == "test-id"


def test_reader_does_not_create_missing_collection(monkeypatch, temp_chroma):
    import context_retrieval as backend

    monkeypatch.setattr(backend, "collection", None)
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: temp_chroma)
    with pytest.raises(NotFoundError):
        backend._ensure_collection()
    assert temp_chroma.list_collections() == []


def qt_selected_model(label):
    from GUI_settings import update_settings_from_ui

    combo = SimpleNamespace(currentText=lambda: "")
    spin = SimpleNamespace(value=lambda: 5)
    ui = SimpleNamespace(
        embedding_model_combo=SimpleNamespace(currentData=lambda: label, currentText=lambda: label),
        groq_model_combo=combo, groq_rewrite_combo=combo, ollama_model_combo=combo,
        chromadb_path_edit=SimpleNamespace(text=lambda: "synthetic-only"),
        collection_combo=SimpleNamespace(currentText=lambda: "synthetic_test"),
        system_message_edit=SimpleNamespace(toPlainText=lambda: ""),
        top_k_spin=spin, alpha_spin=spin, beta_spin=spin, gamma_spin=spin,
        lambda_mmr_spin=spin, tts_speed_spin=spin, tts_volume_spin=spin,
        tts_lang_combo=combo, tts_tld_combo=combo,
    )
    settings = {"generation_provider": "ollama"}
    update_settings_from_ui(ui, settings)
    assert settings["embedding_model"] == label
    return settings


@pytest.mark.parametrize("mode", ["phrase", "hybrid", "self_test"])
@pytest.mark.parametrize("state", ["match", "unreadable", "unlabelled", "same-dimension", "different-dimension", "empty"])
def test_direct_gui_checks_selected_model(monkeypatch, capsys, mode, state):
    import GUI_direct_search as direct

    settings = qt_selected_model("other-model" if "dimension" in state else MODEL)
    metadata = {"embedding_model": MODEL} if state == "match" or "dimension" in state else {}
    target = synthetic_collection(metadata, 0 if state == "empty" else 1)
    if state == "unreadable":
        target.count.side_effect = RuntimeError("count unavailable")
    embed = stub_embeddings(monkeypatch, dimensions=3 if state == "different-dimension" else 2)
    client = Mock()
    client.get_collection.return_value = target
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: client)
    args = dict(
        path="synthetic-only", col_name=target.name, query="needle", rewritten_input="needle",
        synonym_dict={}, embedding_model=settings["embedding_model"], top_k=5, alpha=.5,
        beta=.2, gamma=.2, delta=.1, mode=mode, include=["documents", "metadatas", "distances"],
        module_path=os.path.dirname(direct.__file__),
    )

    def in_process(argv, **kwargs):
        for key, value in (kwargs.get("env") or {}).items():
            if key == "_DS_ARGS":
                monkeypatch.setenv(key, value)
        try:
            exec(compile(argv[2], "direct-search-subprocess", "exec"), {})
        except SystemExit as exit_:
            assert exit_.code == 0
        return SimpleNamespace(stdout=capsys.readouterr().out, stderr="", returncode=0)

    def execute():
        if mode == "self_test":
            monkeypatch.setattr(subprocess, "run", in_process)
            window = SimpleNamespace(settings=settings, collection=target)
            return direct.DirectSearchMixin._query_via_subprocess(window, "needle")
        return in_process([sys.executable, "-c", direct._SEARCH_SCRIPT],
                          env={"_DS_ARGS": json.dumps(args)})

    if state == "match":
        with pytest.raises(VectorQueryReached):
            execute()
        embed.assert_called_once_with(model=MODEL, prompt="needle", keep_alive=-1)
        target.query.assert_called_once()
    elif state == "empty":
        output = execute()
        if mode == "self_test":
            assert output["documents"] == [[]]
        else:
            assert json.loads(output.stdout) == {"results": []}
        embed.assert_not_called()
        target.query.assert_not_called()
    else:
        with pytest.raises((RuntimeError, ValueError), match="embedding_model|indexed with"):
            execute()
        embed.assert_not_called()
        target.query.assert_not_called()


@pytest.mark.parametrize("state", ["same-dimension", "different-dimension", "unlabelled", "unreadable", "empty", "match"])
def test_gui_chat_and_helper_reject_incompatible_model(monkeypatch, capsys, state):
    import GUI_chromadb_helper as helper
    import GUI_context as gui

    settings = qt_selected_model(MODEL if state in {"match", "empty"} else "other-model")
    metadata = {} if state in {"unlabelled", "unreadable", "empty"} else {"embedding_model": MODEL}
    target = synthetic_collection(metadata, 0 if state == "empty" else 1)
    if state == "unreadable":
        target.count.side_effect = RuntimeError("count unavailable")
    if state == "match":
        target.query.side_effect = None
        target.query.return_value = {"documents": [["needle"]], "metadatas": [[{"file_name": "a"}]], "distances": [[0.0]]}
    embed = stub_embeddings(monkeypatch, dimensions=3 if state == "different-dimension" else 2)
    client = Mock()
    client.get_collection.return_value = target
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: client)

    def run(args, **kwargs):
        import io
        capsys.readouterr()
        monkeypatch.setattr(sys, "stdin", io.StringIO(kwargs["input"]))
        helper.main()
        return SimpleNamespace(stdout=capsys.readouterr().out, stderr="", returncode=0)

    monkeypatch.setattr(gui.subprocess, "run", run)
    if state in {"empty", "match"}:
        results = gui.get_relevant_context_hybrid(settings, "needle")
        assert bool(results) == (state == "match")
        assert embed.call_count == target.query.call_count == int(state == "match")
    else:
        with pytest.raises(Exception, match="embedding_model|indexed with"):
            gui.get_relevant_context_hybrid(settings, "needle")
        embed.assert_not_called()
        target.query.assert_not_called()


@pytest.mark.parametrize("state", ["unlabelled", "unreadable", "same-dimension", "different-dimension", "empty", "match"])
def test_gui_search_worker_checks_provenance(monkeypatch, state):
    from GUI_workers import ChromaDBSearchWorker

    metadata = {"embedding_model": MODEL if state == "match" else "other-model"}
    if state in {"unlabelled", "unreadable", "empty"}:
        metadata = {}
    target = synthetic_collection(metadata, 0 if state == "empty" else 1)
    if state == "unreadable":
        target.count.side_effect = RuntimeError("count unavailable")
    embed = stub_embeddings(monkeypatch, dimensions=3 if state == "different-dimension" else 2)
    worker = ChromaDBSearchWorker("needle", target, MODEL)
    errors = []
    results = []
    worker.error_occurred.connect(errors.append)
    worker.results_ready.connect(results.append)
    if state == "match":
        with pytest.raises(VectorQueryReached):
            worker.run()
        embed.assert_called_once()
        target.query.assert_called_once()
    else:
        worker.run()
        embed.assert_not_called()
        target.query.assert_not_called()
        if state == "empty":
            assert results == [[]] and not errors
        else:
            assert not results and errors


@pytest.fixture
def watcher_import(monkeypatch):
    import nltk
    monkeypatch.setattr(nltk, "download", lambda *_, **__: None)
    monkeypatch.setitem(sys.modules, "Semantic_chunking", SimpleNamespace(
        extract_text_from_html=lambda _: "fixture", split_into_chunks=lambda text: [{"text": text}],
        generate_chunk_id=lambda _: "test-id",
    ))
    monkeypatch.setitem(sys.modules, "Generate_embeddings", SimpleNamespace(check_gpu_temperature=lambda: None))
    monkeypatch.delitem(sys.modules, "monitor_file_changes_update_chromaDB", raising=False)
    return lambda: importlib.import_module("monitor_file_changes_update_chromaDB")


@pytest.mark.parametrize("count_fails", [False, True])
def test_watcher_leaves_unlabelled_collection_intact(monkeypatch, tmp_path, watcher_import, count_fails):
    target = synthetic_collection({"kept": "value"}, count=0)
    if count_fails:
        target.count.side_effect = RuntimeError("count unavailable")
    client = Mock()
    client.get_collection.return_value = client.get_or_create_collection.return_value = target
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: client)
    embed = stub_embeddings(monkeypatch)
    file = tmp_path / "fixture.html"
    file.write_text("synthetic", encoding="utf-8")
    with pytest.raises(ValueError, match="embedding_model"):
        watcher = watcher_import()
        monkeypatch.setattr(watcher, "check_existing_chunks", lambda _, ids: ids)
        watcher.FileChangeHandler.process_file(SimpleNamespace(_update_vault_json=Mock()), str(file))
    assert target.metadata == {"kept": "value"}
    target.modify.assert_not_called()
    target.add.assert_not_called()
    embed.assert_not_called()
    assert not client.create_collection.called


def csv_fixture(path, rows, *, model=MODEL, dimensions=2):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "file_name", "modification_time", "embedding"])
        writer.writerows(rows)
    manifest = path.with_suffix(".csv.provenance.json")
    manifest.write_text(json.dumps({
        "schema_version": 1, "embedding_model": model, "dimensions": dimensions,
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return manifest


@pytest.mark.parametrize("defect", ["missing", "model", "digest", "shape", "nonfinite", "malformed"])
def test_csv_preflight_precedes_destination_writes(monkeypatch, tmp_path, defect):
    import Text_embeddings_to_chromadb_python as importer

    path = tmp_path / "fixture.csv"
    vector = {"shape": "[0, 1, 0]", "nonfinite": "[0, NaN]"}.get(defect, "[0, 1]")
    manifest = csv_fixture(
        path, [("a", "needle", "a.txt", 1, "[1, 0]"), ("b", "needle", "b.txt", 2, vector)],
        model="other-model" if defect == "model" else MODEL,
    )
    if defect == "missing":
        manifest.unlink()
    elif defect == "digest":
        with path.open("ab") as stream:
            stream.write(b"\n")
    elif defect == "malformed":
        manifest.write_text("invalid JSON", encoding="utf-8")
    target = synthetic_collection({"embedding_model": MODEL})
    init = Mock(return_value=target)
    monkeypatch.setattr(importer, "init_chromadb", init)
    monkeypatch.setattr(importer, "csv_path", str(path))
    monkeypatch.setattr(importer.winsound, "Beep", Mock())

    with pytest.raises(ValueError):
        importer.main()
    init.assert_not_called()
    target.add.assert_not_called()
    with pytest.raises(ValueError):
        importer.insert_batches_to_chromadb(target, path, batch_size=1)
    target.add.assert_not_called()


@pytest.mark.parametrize("count_failure", [False, True])
def test_csv_cannot_relabel_existing_destination(monkeypatch, tmp_path, count_failure):
    import Text_embeddings_to_chromadb_python as importer

    path = tmp_path / "fixture.csv"
    csv_fixture(path, [("a", "needle", "a.txt", 1, "[1, 0]")])
    target = synthetic_collection({"kept": "value"}, count=0)
    if count_failure:
        target.count.side_effect = RuntimeError("count unavailable")
    client = Mock()
    client.get_collection.return_value = client.get_or_create_collection.return_value = target
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: client)
    monkeypatch.setattr(importer, "csv_path", str(path))
    monkeypatch.setattr(importer.winsound, "Beep", Mock())
    with pytest.raises(ValueError, match="embedding_model"):
        importer.main()
    target.modify.assert_not_called()
    target.add.assert_not_called()


@pytest.mark.parametrize("existing", [True, False])
def test_csv_matching_provenance_imports_into_synthetic_collection(monkeypatch, tmp_path, temp_chroma, existing):
    import Text_embeddings_to_chromadb_python as importer

    path = tmp_path / "fixture.csv"
    csv_fixture(path, [("stable-id", "needle", "a.txt", 1, "[1, 0]")])
    if existing:
        temp_chroma.create_collection(
            importer.collection_name, embedding_function=None,
            metadata={"embedding_model": MODEL, "keep": "existing"},
        )
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: temp_chroma)
    monkeypatch.setattr(importer, "csv_path", str(path))
    monkeypatch.setattr(importer.winsound, "Beep", Mock())
    importer.main()
    collection = temp_chroma.get_collection(importer.collection_name, embedding_function=None)
    assert collection.get()["ids"] == ["stable-id"]
    assert collection.metadata == (
        {"embedding_model": MODEL, "keep": "existing"} if existing else {"embedding_model": MODEL}
    )


def test_csv_never_writes_to_incompatible_destination(tmp_path):
    import Text_embeddings_to_chromadb_python as importer

    path = tmp_path / "fixture.csv"
    csv_fixture(path, [("a", "needle", "a.txt", 1, "[1, 0]")])
    target = synthetic_collection({"embedding_model": "other-model"})
    with pytest.raises(ValueError, match="other-model"):
        importer.insert_batches_to_chromadb(target, path, batch_size=1)
    target.add.assert_not_called()
