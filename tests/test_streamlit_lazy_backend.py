import sys
from types import SimpleNamespace

from test_generation_streamlit import load_streamlit_app


def test_streamlit_search_mode_does_not_load_the_generation_backend(
    load_streamlit_app, monkeypatch
):
    app, display = load_streamlit_app
    display.selectbox = lambda *args, **kwargs: "offline"
    calls = []
    monkeypatch.setattr(app, "groq_router", lambda: calls.append("groq"))
    monkeypatch.setattr(app, "ollama_router", lambda: calls.append("ollama"))

    assert app._generation_settings() == ("offline", None, None, False)
    assert sys.modules["streamlit_groq_lama_chromadb_RAG_ETTS"] is None
    assert calls == []


def test_streamlit_initializes_the_retrieval_backend_on_first_use(
    load_streamlit_app, monkeypatch
):
    app, _display = load_streamlit_app
    calls = []
    backend = SimpleNamespace(collection=None)

    def initialize():
        calls.append("initialize")
        backend.collection = object()

    def retrieve(prompt, **options):
        calls.append((prompt, options))
        return "", []

    backend.initialize_collection = initialize
    backend.get_relevant_context_hybrid = retrieve
    monkeypatch.setitem(sys.modules, "streamlit_groq_lama_chromadb_RAG_ETTS", backend)

    app.get_relevant_context_hybrid("first", rewrite=False)
    app.get_relevant_context_hybrid("next", rewrite=False)

    assert calls == ["initialize", ("first", {"rewrite": False}), ("next", {"rewrite": False})]
