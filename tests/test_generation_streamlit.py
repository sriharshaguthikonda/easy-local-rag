import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class Session(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class Display:
    def __init__(self):
        self.session_state = Session(conversation_history=[], tts_enabled=False)
        self.errors = []
        self.captions = []
        self.markdowns = []

    def empty(self):
        return self

    def markdown(self, text, **options):
        self.markdowns.append(text)

    def set_page_config(self, **options):
        pass

    def info(self, message):
        pass

    def caption(self, message):
        self.captions.append(message)

    def error(self, message):
        self.errors.append(message)


@pytest.fixture
def load_streamlit_app(monkeypatch):
    display = Display()
    display.cache_resource = lambda function: function
    monkeypatch.setitem(sys.modules, "streamlit", display)
    monkeypatch.setitem(sys.modules, "streamlit_groq_lama_chromadb_RAG_ETTS", None)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", app_path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app, display


@pytest.mark.parametrize("listed_preference", [True, False])
def test_reported_404_recovers_with_exact_numbered_evidence(
    load_streamlit_app, monkeypatch, listed_preference
):
    import httpx
    from groq import Groq

    app, display = load_streamlit_app
    evidence = "Synthetic evidence [alpha], right here."
    preferred = "llama-3.3-70b-versatile"
    replacement = "openai/gpt-oss-120b"
    answer_text = "The evidence supports this [1]."
    events = []
    requests = []

    def provider(request):
        if request.url.path.endswith("/models"):
            events.append("catalogue")
            model_ids = [preferred, replacement] if listed_preference else [replacement]
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": model, "active": True} for model in model_ids
            ]})
        assert request.url.path.endswith("/chat/completions")
        parameters = json.loads(request.content)
        model = parameters["model"]
        events.append(model)
        requests.append(parameters)
        if model == preferred:
            return httpx.Response(404, json={"error": {
                "message": "Model not found", "code": "model_not_found",
            }})
        chunk = {"model": model, "choices": [{"delta": {"content": answer_text}}]}
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    def deny_socket(*args, **kwargs):
        pytest.fail("The application tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_socket)
    http_client = httpx.Client(transport=httpx.MockTransport(provider))
    client = Groq(api_key="fixture-only", http_client=http_client)
    router = app.groq_router(client=client)
    router.catalogue()
    events.clear()
    retrievals = []

    def retrieve(prompt, **options):
        retrievals.append(prompt)
        assert options["rewrite"] is True
        return evidence, [{"file_name": "sample.txt", "document": evidence}]

    context, answer, sources = app.chat_with_model(
        "Question", "System", preferred=preferred, router=router, retrieve=retrieve,
    )

    assert (context, answer) == (evidence, answer_text)
    assert [(row["citation_id"], row["document"]) for row in sources] == [(1, evidence)]
    assert retrievals == ["Question"]
    assert events == [preferred, "catalogue", replacement]
    assert requests[0]["messages"] == requests[1]["messages"]
    assert [request["service_tier"] for request in requests] == ["on_demand"] * 2
    assert requests[1]["include_reasoning"] is False
    assert requests[1]["reasoning_effort"] == "low"
    assert requests[0]["messages"][-1]["content"].count(evidence) == 1
    assert replacement in display.captions[-1]
    assert preferred in display.captions[-1]
    assert "unavailable" in display.captions[-1].lower()
    assert len(display.session_state.conversation_history) == 2
    client.close()


def test_local_generation_failure_preserves_sources_and_skips_cloud_rewrite(load_streamlit_app):
    app, display = load_streamlit_app
    evidence = "One local document."
    model_error = app.GenerationError("no_candidate", "No compatible model is available.")

    def retrieve(prompt, **options):
        assert options["rewrite"] is False
        return evidence, [{"file_name": "local.txt", "document": evidence}]

    def generate(**request):
        raise model_error

    context, answer, sources = app.chat_with_model(
        "Question",
        "System",
        provider="ollama",
        preferred="phi3:latest",
        router=SimpleNamespace(generate=generate),
        retrieve=retrieve,
    )

    assert (context, answer) == (evidence, None)
    assert sources[0]["document"] == evidence
    assert display.errors == ["No compatible model is available."]
    assert not display.session_state.conversation_history



def test_menu_refresh_keeps_saved_preference_and_separates_providers(
    load_streamlit_app, monkeypatch
):
    app, display = load_streamlit_app
    app._initialize_session()
    monkeypatch.setattr(app, "load_dotenv", lambda: None)
    display.session_state.generation_model_groq = "retired"
    refreshes = []
    menus = []
    models = ["retired", "replacement"]

    def catalogue(*, refresh):
        refreshes.append(refresh)
        return [SimpleNamespace(id=model) for model in models]

    def selectbox(label, options, **settings):
        if label == "Provider / mode":
            return display.session_state.generation_provider
        menus.append([settings["format_func"](model) for model in options])
        return options[0]

    display.selectbox = selectbox
    display.button = lambda *args, **options: bool(refreshes)
    display.checkbox = lambda *args, **options: False
    monkeypatch.setattr(app, "groq_router", lambda: SimpleNamespace(catalogue=catalogue))
    monkeypatch.setattr(app, "ollama_router", lambda: SimpleNamespace(
        catalogue=lambda **options: [SimpleNamespace(id="local")]
    ))

    assert app._generation_settings()[2] == "retired"
    models.remove("retired")
    assert app._generation_settings()[2] == "retired"
    display.session_state.generation_provider = "ollama"
    assert app._generation_settings()[0:3:2] == ("ollama", "phi-3")

    assert refreshes == [False, True]
    assert menus[0] == ["retired", "replacement"]
    assert menus[1] == ["retired (saved, unavailable)", "replacement"]
    assert menus[2] == ["phi-3 (saved, unavailable)", "local"]
    assert display.session_state.generation_model_groq == "retired"


def test_partial_stream_keeps_evidence_and_never_commits_chat_or_speech(
    load_streamlit_app, monkeypatch
):
    app, display = load_streamlit_app
    display.session_state.tts_enabled = True
    monkeypatch.setattr(app, "ensure_tts_worker", lambda: pytest.fail("Speech started"))

    def generate(**request):
        assert request["strict"] is True
        assert request["cancelled"]() is False
        request["on_chunk"]("Partial response")
        raise app.GenerationError("cancelled", "Generation cancelled.")

    context, answer, sources = app.chat_with_model(
        "question", "system", strict=True, cancelled=lambda: False,
        router=SimpleNamespace(generate=generate),
        retrieve=lambda *args, **options: ("evidence", [{"document": "evidence"}]),
    )

    assert (context, answer) == ("evidence", None)
    assert sources[0]["document"] == "evidence"
    assert display.errors == ["Generation cancelled."]
    assert not display.session_state.conversation_history

def test_streamlit_offline_entry_runs_local_search_without_provider_or_speech(
    load_streamlit_app, monkeypatch
):
    from contextlib import nullcontext

    app, display = load_streamlit_app
    evidence = "Synthetic pseudoferrite record."
    collection = SimpleNamespace(get=lambda **options: {
        "ids": ["one"], "documents": [evidence], "metadatas": [{}],
    })
    monkeypatch.setattr(app, "open_local_collection", lambda *args: collection)

    def unexpected(*args, **kwargs):
        pytest.fail("Offline search attempted to initialize a model or open a socket")

    monkeypatch.setattr(app, "groq_router", unexpected)
    monkeypatch.setattr(app, "ollama_router", unexpected)
    monkeypatch.setattr(app, "_require_rag_backend", unexpected)
    monkeypatch.setattr(app, "ensure_tts_worker", unexpected)
    monkeypatch.setattr(app, "load_dotenv", unexpected)
    monkeypatch.setattr(socket.socket, "connect", unexpected)

    display.sidebar = nullcontext()
    display.sidebar.title = lambda *args, **kwargs: None
    display.sidebar.expander = lambda *args, **kwargs: nullcontext()
    display.columns = lambda widths: tuple(nullcontext() for _ in widths)
    display.selectbox = lambda label, choices, **kwargs: (
        "offline" if label == "Provider / mode" else choices[0]
    )
    display.chat_input = lambda *args, **kwargs: "pseudoferrite"
    display.chat_message = lambda *args, **kwargs: nullcontext()
    display.expander = lambda *args, **kwargs: nullcontext()
    display.slider = lambda label, minimum, maximum, default, **kwargs: default
    for name in ("title", "text_area", "multiselect", "date_input", "text_input",
                 "container", "button", "metric", "file_uploader"):
        setattr(display, name, lambda *args, **kwargs: None)

    app.main()

    assert evidence in display.markdowns
    assert display.session_state.current_sources[0]["document"] == evidence
    assert display.session_state.conversation_history == []
    assert not display.errors
