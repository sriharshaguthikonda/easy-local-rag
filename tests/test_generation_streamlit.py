import importlib.util
import json
import socket
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_provider_admission")

from structural_answer_fixtures import completed_excerpt, displayed_excerpt, excerpt_payload


class Session(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class Display:
    def __init__(self):
        self.session_state = Session(conversation_history=[], tts_enabled=False)
        self.errors = []
        self.warnings = []
        self.captions = []
        self.markdowns = []
        self.plain_text = []

    def empty(self):
        return self

    def markdown(self, text, **options):
        self.markdowns.append(text)

    def text(self, text):
        self.plain_text.append(text)

    def set_page_config(self, **options):
        pass

    def info(self, message):
        pass

    def caption(self, message):
        self.captions.append(message)

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def setup_main(self, provider, prompt=None):
        self.sidebar = nullcontext()
        self.sidebar.title = lambda *args, **kwargs: None
        self.sidebar.expander = lambda *args, **kwargs: nullcontext()
        self.columns = lambda widths: tuple(nullcontext() for _ in widths)
        self.selectbox = lambda label, choices, **kwargs: (
            provider if label == "Provider / mode" else choices[0]
        )
        self.chat_input = lambda *args, **kwargs: prompt
        self.chat_message = lambda *args, **kwargs: nullcontext()
        self.expander = lambda *args, **kwargs: nullcontext()
        self.slider = lambda label, minimum, maximum, default, **kwargs: default
        for name in ("title", "text_area", "multiselect", "date_input", "text_input",
                     "button", "metric", "file_uploader", "image"):
            setattr(self, name, lambda *args, **kwargs: None)


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


@pytest.mark.parametrize("legacy_default", [False, True])
def test_streamlit_requires_explicit_cloud_speech_consent(load_streamlit_app, monkeypatch, legacy_default):
    app, display = load_streamlit_app
    display.session_state.pop("tts_enabled")
    if legacy_default:
        display.session_state.tts_enabled = True
    app._initialize_session()
    spoken = []
    monkeypatch.setattr(
        app, "ensure_tts_worker", lambda: {"queue": SimpleNamespace(put=spoken.append)}
    )
    evidence = "Synthetic evidence."
    _, answer, _ = app.chat_with_model(
        "Original question", "System",
        router=SimpleNamespace(generate=lambda **_: completed_excerpt(evidence)),
        retrieve=lambda *_a, **_k: (evidence, [{"document": evidence}]),
    )

    assert answer == displayed_excerpt(evidence)
    assert spoken == []


@pytest.mark.parametrize("provider", ["groq", "ollama"])
@pytest.mark.parametrize("background", [False, True])
def test_streamlit_guarded_requests_keep_history_and_current_sources_separate(
    load_streamlit_app, provider, background
):
    from rag_prompting import CONTEXT_GUARD

    app, display = load_streamlit_app
    requests = []

    def retrieve(question, **options):
        doc = f"Synthetic evidence for {question} only."
        return doc, [{"file_name": "fixture.txt", "document": doc}]

    def generate(**request):
        requests.append(request)
        cue = ("first cue", "second cue")[len(requests) - 1]
        return completed_excerpt(f"Synthetic evidence for {cue} only.")

    for question in ("first cue", "second cue"):
        options = dict(provider=provider, router=SimpleNamespace(generate=generate), retrieve=retrieve)
        if background:
            job = app.start_generation_job(question, "Trusted policy.", **options)
            job.thread.join(timeout=5)
            assert not job.thread.is_alive()
            app._generation_observer_impl()
        else:
            _context, answer, _sources = app.chat_with_model(question, "Trusted policy.", **options)
            assert answer == displayed_excerpt(f"Synthetic evidence for {question} only.")

    first, second = [request["messages"] for request in requests]
    assert second[1:3] == [
        {"role": "user", "content": "first cue"},
        {"role": "assistant", "content": "A previous turn displayed quotations from its sources."},
    ]
    assert "Synthetic evidence for first cue only." not in str(second)
    assert "Synthetic evidence for second cue only." in second[-1]["content"]
    assert first[0]["role"] == second[0]["role"] == "system"
    assert first[0]["content"].startswith("Trusted policy.")
    assert CONTEXT_GUARD in first[0]["content"]
    assert CONTEXT_GUARD not in first[-1]["content"]
    assert all(request["text_only"] for request in requests)


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
    answer_text = displayed_excerpt(evidence)
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
        chunk = {"model": model, "choices": [{"index": 0, "delta": {"content": excerpt_payload(evidence)}}]}
        finished = {"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(finished)}\n\ndata: [DONE]\n\n",
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
    assert display.errors == ["Generation failed."]
    assert display.session_state.last_answer_result.status == "provider_failed"
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
    display.session_state.allow_cloud_speech = True
    monkeypatch.setattr(app, "ensure_tts_worker", lambda: pytest.fail("Speech started"))

    def generate(**request):
        assert request["strict"] is True
        assert request["cancelled"]() is False
        assert "on_chunk" not in request
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


def test_background_generation_cancel_reaches_blocked_provider_without_commit(
    load_streamlit_app, monkeypatch
):
    app, display = load_streamlit_app
    app._initialize_session()
    display.session_state.allow_cloud_speech = True
    started = threading.Event()
    wake = threading.Event()
    calls = []
    spoken = []
    monkeypatch.setattr(
        app,
        "ensure_tts_worker",
        lambda: {"queue": SimpleNamespace(put=spoken.append)},
    )

    def generate(**request):
        calls.append(request)
        started.set()
        while not request["cancelled"]():
            wake.wait(0.01)
        return completed_excerpt("evidence")

    router = SimpleNamespace(generate=generate)
    job = app.start_generation_job(
        "question",
        "system",
        provider="groq",
        preferred="selected",
        router=router,
        retrieve=lambda *args, **options: (
            "evidence",
            [{"file_name": "source.txt", "document": "evidence"}],
        ),
    )

    assert started.wait(2)
    display.button = lambda *args, on_click, **kwargs: on_click()
    app._generation_observer_impl()
    job.thread.join(timeout=3)
    assert not job.thread.is_alive()

    # Completion is finalized by the UI/run thread, never by the worker.
    app._generation_observer_impl()
    assert len(calls) == 1
    assert not display.session_state.conversation_history
    assert spoken == []
    assert display.errors == ["Generation cancelled."]
    assert display.session_state.generation_job is None


def test_selected_router_and_strict_policy_cross_retrieval_boundary(
    load_streamlit_app,
):
    app, _display = load_streamlit_app
    retrieval_calls = []
    router = SimpleNamespace(generate=lambda **request: completed_excerpt("evidence"))

    def retrieve(prompt, **options):
        retrieval_calls.append((prompt, options))
        return "evidence", [{"file_name": "source.txt", "document": "evidence"}]

    app.chat_with_model(
        "question",
        "system",
        preferred="selected",
        router=router,
        strict=True,
        retrieve=retrieve,
    )

    assert retrieval_calls[0][1]["router"] is router
    assert retrieval_calls[0][1]["strict"] is True

@pytest.mark.parametrize("quote", [
    "![synthetic](https://example.invalid/answer-marker) [77].",
    "A quotation with no embedded citation.",
])
def test_accepted_background_answer_and_sources_survive_full_rerun(
    load_streamlit_app, monkeypatch, quote
):
    app, display = load_streamlit_app
    app._initialize_session()
    reruns = []
    display.rerun = lambda: reruns.append(True)
    answer = displayed_excerpt(quote)
    router = SimpleNamespace(generate=lambda **request: completed_excerpt(
        quote, actual_model="local-model",
    ))
    job = app.start_generation_job(
        "Question", "System", provider="ollama", preferred="local-model", router=router,
        retrieve=lambda *args, **options: (quote, [{"file_name": "source.txt", "document": quote}]),
    )
    job.thread.join(timeout=3)
    assert not job.thread.is_alive()
    app._generation_observer_impl()
    assert reruns == [True]
    assert display.session_state.generation_job is None

    display.markdowns.clear()
    display.plain_text.clear()
    display.setup_main("ollama")
    monkeypatch.setattr(app, "_generation_settings", lambda **kwargs: (
        "ollama", router, "local-model", True
    ))
    monkeypatch.setattr(app, "generate_word_cloud", lambda text: SimpleNamespace(to_array=lambda: []))
    plotly = ModuleType("plotly")
    plotly.express = ModuleType("plotly.express")
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.express", plotly.express)
    def unexpected_html(*args, **kwargs):
        pytest.fail("A completed chat rerun embedded third-party HTML")

    display.components = SimpleNamespace(v1=SimpleNamespace(html=unexpected_html))
    rendered = []
    monkeypatch.setattr(app, "render_source_documents", lambda sources, ids: rendered.append((
        [source["document"] for source in sources], ids
    )))

    app.main()

    assert "Question" in display.plain_text
    assert answer in display.plain_text
    assert answer not in display.markdowns
    assert rendered == [([quote], [1])]
    assert display.warnings == []


def test_streamlit_offline_entry_runs_local_search_without_provider_or_speech(
    load_streamlit_app, monkeypatch
):
    app, display = load_streamlit_app
    evidence = "Synthetic pseudoferrite record."
    collection = SimpleNamespace(id="synthetic-collection-id", count=lambda: 1, get=lambda **options: {
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

    display.setup_main("offline", "pseudoferrite")

    app.main()

    assert evidence in display.plain_text
    assert display.session_state.current_sources[0]["document"] == evidence
    assert display.session_state.current_sources[0]["collection_id"] == "synthetic-collection-id"
    assert display.session_state.current_sources[0]["chunk_id"] == "one"
    assert display.session_state.conversation_history == []
    assert not display.errors

@pytest.mark.parametrize("turn_index", [0, -1])
def test_streamlit_stops_if_budget_would_change_system_or_current_evidence(
    load_streamlit_app, monkeypatch, turn_index,
):
    app, display = load_streamlit_app

    original_prepare = app._prepare_generation

    def prepare_with_truncated_request(*args, **options):
        prepared = original_prepare(*args, **options)
        messages = prepared.answer.messages()
        messages[turn_index]["content"] = "Truncated synthetic request."
        return prepared._replace(request={**prepared.request, "messages": messages})

    def fail_generate(**request):
        pytest.fail("A truncated request reached generation")

    monkeypatch.setattr(app, "_prepare_generation", prepare_with_truncated_request)
    context, answer, sources = app.chat_with_model(
        "Question", "System", provider="ollama", router=SimpleNamespace(generate=fail_generate),
        retrieve=lambda *args, **options: ("evidence", [{"document": "evidence"}]),
    )
    assert context == "evidence"
    assert answer is None
    assert sources[0]["document"] == "evidence"
    assert display.errors == ["The full evidence does not fit the input budget."]
