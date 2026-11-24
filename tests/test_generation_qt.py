"""Network-free Qt integration tests for issue #44."""

from types import SimpleNamespace

import pytest

PyQt5 = pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QComboBox, QCheckBox, QLineEdit, QTextEdit

import GUI_context
import GUI_settings
from GUI_chat import ChatFunctionalityMixin
from GUI_models import ModelLoadingMixin
from GUI_workers import ChatWorker


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


class StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, timeout=None):
        self.messages.append((message, timeout))


class ModelWindow(ModelLoadingMixin):
    def __init__(self, router):
        self.settings = dict(GUI_settings.DEFAULT_SETTINGS)
        self.settings["generation_router_groq"] = router
        self.groq_model_combo = QComboBox()
        self.groq_rewrite_combo = QComboBox()
        self.ollama_model_combo = QComboBox()
        self.statusBar = StatusBar()


class FakeCatalogueRouter:
    def __init__(self, records):
        self.records = records
        self.refreshes = []

    def catalogue(self, refresh=False):
        self.refreshes.append(refresh)
        return list(self.records)


def test_model_refresh_keeps_missing_saved_preference_and_reports_it(qapp):
    router = FakeCatalogueRouter(
        [SimpleNamespace(id="available-model", capabilities={"stream": True})]
    )
    window = ModelWindow(router)
    window.settings["groq_model"] = "retired-model"
    window.groq_model_combo.addItem("retired-model")
    window.groq_model_combo.setCurrentText("retired-model")

    window.load_groq_models(refresh=True)

    assert window.groq_model_combo.currentText() == "retired-model (unavailable)"
    assert "unavailable" in window.statusBar.messages[-1][0].lower()
    assert router.refreshes == [True]


def test_settings_round_trip_persists_provider_and_strict_pin(tmp_path, monkeypatch, qapp):
    window = SimpleNamespace(
        embedding_model_combo=QComboBox(),
        groq_model_combo=QComboBox(),
        groq_rewrite_combo=QComboBox(),
        ollama_model_combo=QComboBox(),
        generation_provider_combo=QComboBox(),
        strict_generation_model_checkbox=QCheckBox(),
        chromadb_path_edit=QLineEdit(),
        collection_combo=QComboBox(),
        system_message_edit=QTextEdit(),
        top_k_spin=SimpleNamespace(value=lambda: 5),
        alpha_spin=SimpleNamespace(value=lambda: 0.5),
        beta_spin=SimpleNamespace(value=lambda: 0.3),
        gamma_spin=SimpleNamespace(value=lambda: 0.2),
        lambda_mmr_spin=SimpleNamespace(value=lambda: 0.5),
        tts_speed_spin=SimpleNamespace(value=lambda: 1.4),
        tts_volume_spin=SimpleNamespace(value=lambda: 0.5),
        tts_lang_combo=QComboBox(),
        tts_tld_combo=QComboBox(),
        statusBar=StatusBar(),
    )
    window.generation_provider_combo.addItems(["groq", "ollama"])
    window.generation_provider_combo.setCurrentText("ollama")
    window.strict_generation_model_checkbox.setChecked(True)
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(GUI_settings, "_settings_path", lambda: settings_path)
    settings = GUI_settings.DEFAULT_SETTINGS.copy()

    GUI_settings.save_settings(window, settings)

    assert settings["generation_provider"] == "ollama"
    assert settings["strict_generation_model"] is True
    assert '"generation_provider": "ollama"' in settings_path.read_text(encoding="utf-8")


def test_chat_worker_emits_fallback_model_status_and_stream(qapp):
    events = []

    class Router:
        def generate(self, **kwargs):
            assert kwargs["preferred"] == "saved-model"
            assert kwargs["strict"] is False
            kwargs["on_chunk"]("answer from fallback")
            return SimpleNamespace(
                text="answer from fallback",
                preferred_model="saved-model",
                actual_model="replacement-model",
                reason="preferred unavailable",
            )

    settings = dict(GUI_settings.DEFAULT_SETTINGS)
    settings.update(
        {
            "generation_provider": "groq",
            "groq_model": "saved-model",
            "generation_router_groq": Router(),
            "strict_generation_model": False,
        }
    )
    worker = ChatWorker(
        "question",
        settings,
        [{"id": "doc-1", "meta": {"file_name": "doc.txt"}, "document": "evidence"}],
        [],
        router=settings["generation_router_groq"],
    )
    worker.response_chunk.connect(lambda chunk: events.append(("chunk", chunk)))
    worker.status_update.connect(lambda status: events.append(("status", status)))
    worker.response_complete.connect(lambda text: events.append(("complete", text)))

    worker.run()

    assert ("chunk", "answer from fallback") in events
    assert any("replacement-model" in value for kind, value in events if kind == "status")
    assert ("complete", "answer from fallback") in events


def test_chat_worker_cancels_between_stream_chunks_without_committing_history(qapp):
    events = []
    holder = {}

    class Router:
        def generate(self, **kwargs):
            kwargs["on_chunk"]("first chunk")
            holder["worker"].request_cancel()
            return SimpleNamespace(
                text="must not commit",
                preferred_model="saved-model",
                actual_model="saved-model",
                reason="",
            )

    settings = dict(GUI_settings.DEFAULT_SETTINGS)
    settings.update(
        {
            "generation_provider": "groq",
            "groq_model": "saved-model",
            "strict_generation_model": False,
        }
    )
    history = []
    worker = ChatWorker(
        "question",
        settings,
        [{"id": "doc-1", "meta": {"file_name": "doc.txt"}, "document": "evidence"}],
        history,
        router=Router(),
    )
    holder["worker"] = worker
    worker.status_update.connect(lambda status: events.append(("status", status)))
    worker.response_complete.connect(lambda text: events.append(("complete", text)))

    worker.run()

    assert any(value == "Generation cancelled" for kind, value in events if kind == "status")
    assert ("complete", "") in events
    assert history == []


def test_local_rewrite_uses_original_query_without_groq(qapp, monkeypatch):
    def fail_groq(*args, **kwargs):
        raise AssertionError("local generation must not construct Groq")

    monkeypatch.setattr(GUI_context, "Groq", fail_groq)

    rewritten, keywords = GUI_context.rewrite_input_and_generate_synonyms(
        {"generation_provider": "ollama", "groq_rewrite_model": "retired-model"},
        "original query",
    )

    assert rewritten == "original query"
    assert keywords == {}


def test_groq_rewrite_uses_injected_router_boundary(qapp):
    calls = []

    class Router:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                text='{"rephrased": "rewritten", "keywords": {"term": {"synonyms": ["word"]}}}'
            )

    rewritten, keywords = GUI_context.rewrite_input_and_generate_synonyms(
        {
            "generation_provider": "groq",
            "groq_rewrite_model": "rewrite-model",
            "generation_router_groq": Router(),
        },
        "original",
    )

    assert rewritten == "rewritten"
    assert keywords["term"]["synonyms"] == ["word"]
    assert calls[0]["preferred"] == "rewrite-model"
    assert calls[0]["stream"] is False
    assert calls[0]["response_format"] == {"type": "json_object"}


class SearchWindow(ChatFunctionalityMixin):
    def __init__(self):
        self.settings = dict(GUI_settings.DEFAULT_SETTINGS)
        self.settings["generation_provider"] = "groq"
        self.chat_input = QTextEdit("offline query")
        self.chat_display = QTextEdit()
        self.context_display = QTextEdit()
        self.send_btn = QCheckBox()
        self.search_only_btn = QCheckBox()
        self.stop_btn = QCheckBox()
        self.statusBar = StatusBar()
        self.collection = object()
        self.conversation_history = []
        self.chat_worker = None
        self.tts_enabled = False
        self.results = None

    def update_settings_from_ui(self):
        return None

    def append_message(self, *args, **kwargs):
        return None

    def set_progress_stage(self, stage):
        self.progress_stage = stage

    def get_relevant_context_hybrid(self, query):
        pytest.fail("provider path used")

    def on_context_ready(self, results):
        self.results = results


def test_search_only_uses_offline_documents_without_generation(qapp, monkeypatch):
    window = SearchWindow()
    calls = []

    def search_documents(collection, query, limit):
        calls.append((collection, query, limit))
        return [
            {
                "id": "doc-1",
                "document": "offline answer evidence",
                "metadata": {"file_name": "doc.txt"},
                "relevance_score": 0.8,
            }
        ]

    monkeypatch.setattr("offline_search.search_documents", search_documents)
    monkeypatch.setattr(window, "get_relevant_context_hybrid", lambda query: pytest.fail("provider path used"))

    window.search_only()

    assert calls == [(window.collection, "offline query", 5)]
    assert window.results[0]["document"] == "offline answer evidence"
    assert window.chat_worker is None


@pytest.mark.parametrize("mode, provider, should_rewrite", [
    ("hybrid", "groq", True),
    ("hybrid", "ollama", False),
    ("phrase", "groq", False),
])
def test_direct_search_retains_modes_and_scopes_optional_rewrite(
    qapp, monkeypatch, mode, provider, should_rewrite,
):
    import json
    import GUI_direct_search

    calls = []
    statuses = []
    rows = []

    class Router:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                text='{"rephrased": "expanded", "keywords": {}}',
                provider="groq",
                actual_model="replacement",
                reason="saved model unavailable",
            )

    def fake_subprocess(args, **kwargs):
        payload = json.loads(kwargs["env"]["_DS_ARGS"])
        assert payload["mode"] == mode
        assert payload["query"] == "original"
        assert payload["rewritten_input"] == ("expanded" if should_rewrite else "original")
        assert payload["embedding_model"] == "index-embedding"
        return SimpleNamespace(returncode=0, stdout='{"results": [{"file_name": "source"}]}')

    monkeypatch.setattr(GUI_direct_search.subprocess, "run", fake_subprocess)
    worker = GUI_direct_search.DirectSearchSubprocessWorker(
        query="original", chroma_path="unused", collection_name="unused",
        embed_model="index-embedding", rewrite_model="saved-model", n_results=5,
        include=["documents"], mode=mode, rewrite_settings={
            "generation_provider": provider,
            "generation_router_groq": Router(),
            "strict_generation_model": True,
        },
    )
    worker.status_update.connect(statuses.append)
    worker.results_ready.connect(rows.extend)

    worker.run()

    assert len(calls) == int(should_rewrite)
    assert rows == [{"file_name": "source"}]
    if should_rewrite:
        assert calls[0]["preferred"] == "saved-model"
        assert calls[0]["strict"] is True
        assert statuses == ["Rewrite used groq: replacement (saved model unavailable)"]
    else:
        assert statuses == []
