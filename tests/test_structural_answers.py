"""Synthetic regressions for the maintained Chroma answer boundary."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_provider_admission")

from generation_models import GroqRouter
from test_generation_cli import _load_module
from test_generation_qt import qapp
from test_generation_streamlit import load_streamlit_app

MODEL = "llama-3.3-70b-versatile"
QUESTION = "Which color is recorded for fixture Atlas?"
SOURCE = "Fixture Atlas lamp is red."
OMITTED = "Fixture excluded Atlas lamp is violet."
SURFACES = (
    "qt", "web-sync", "web-background",
    "groq_lama_chromadb_RAG_ETTS.py", "streamlit_groq_lama_chromadb_RAG_ETTS.py",
)


def packet(alias, quote):
    return json.dumps({"excerpts": [{"alias": alias, "quote": quote}]})


def sources():
    return [
        {"collection_id": "synthetic", "chunk_id": f"chunk-{index}",
         "file_name": f"fixture-{index}.txt", "document": document}
        for index, document in enumerate((SOURCE, OMITTED), start=1)
    ]


def groq(chunks, requests):
    def create(**request):
        requests.append(request)
        assert request["stream"] is True
        return chunks()

    return GroqRouter(client=SimpleNamespace(
        models=SimpleNamespace(list=lambda: [{"id": MODEL}]),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))


def delta(text="", finish=None):
    return {"model": MODEL, "choices": [{"delta": {"content": text}, "finish_reason": finish}]}


def exercise(surface, request, monkeypatch, router):
    """Use the actual callers and provider router with inert retrieval and speech."""
    candidates = sources()
    sent = candidates[:1]
    visible, history, spoken = [], [], []
    result = None
    if surface == "qt":
        request.getfixturevalue("qapp")
        from GUI_workers import ChatWorker

        worker = ChatWorker(
            QUESTION,
            {"system_message": "Use fixture evidence.", "groq_model": MODEL,
             "strict_generation_model": True},
            sent, history, router=router,
        )
        results = []
        worker.response_chunk.connect(visible.append)
        worker.generation_result.connect(results.append)
        worker.run()
        result = results[-1] if results else None
    elif surface.startswith("web-"):
        app, display = request.getfixturevalue("load_streamlit_app")
        display.session_state.allow_cloud_speech = True
        monkeypatch.setattr(
            app, "ensure_tts_worker", lambda: {"queue": SimpleNamespace(put=spoken.append)}
        )
        options = {
            "preferred": MODEL, "router": router, "strict": True,
            "retrieve": lambda *_args, **_kwargs: (SOURCE, sent),
        }
        if surface == "web-sync":
            app.chat_with_model(QUESTION, "Use fixture evidence.", **options)
        else:
            job = app.start_generation_job(QUESTION, "Use fixture evidence.", **options)
            job.thread.join(timeout=5)
            assert not job.thread.is_alive()
            app._generation_observer_impl()
        result = display.session_state.get("last_answer_result")
        visible = display.plain_text
        history = display.session_state.conversation_history
    else:
        backend = _load_module(monkeypatch, surface, "structural_cli_under_test")
        monkeypatch.setattr(backend, "get_relevant_context_hybrid", lambda *a, **k: (SOURCE, sent))
        monkeypatch.setattr(backend, "_accepted_speech", spoken.append)
        try:
            result = backend.generate_answer(
                QUESTION, "Use fixture evidence.", MODEL, history,
                router=router, strict=True, on_chunk=visible.append, allow_cloud_speech=True,
            )
        except RuntimeError:
            # Baseline propagates provider errors. The invariant checked below
            # is independent of whether that path raises or returns a result.
            pass
    return visible, history, spoken, result


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(("payload", "finish"), (
    pytest.param("Unsupported compound [99, 100]", "stop", id="compound"),
    pytest.param("Unsupported nonexistent [99]", "stop", id="plain-unknown"),
    pytest.param(packet(99, SOURCE), "stop", id="unknown-alias"),
    pytest.param(packet(2, OMITTED), "stop", id="excluded-alias"),
    pytest.param(packet(1, OMITTED), "stop", id="wrong-source-quotation"),
    pytest.param(packet(1, "Fixture Atlas lamp is glowing red."), "stop", id="fabricated-quotation"),
    pytest.param('{"excerpts": [', "stop", id="malformed-json"),
    pytest.param(packet(True, SOURCE), "stop", id="invalid-field-type"),
    pytest.param("", "stop", id="empty"),
    pytest.param(packet(1, SOURCE), "length", id="output-limit"),
    pytest.param(packet(1, SOURCE), None, id="missing-terminal"),
))
def test_rejected_answer_stays_out_of_normal_sinks(surface, payload, finish, request, monkeypatch):
    requests = []

    def chunks():
        yield delta(payload)
        if finish is not None:
            yield delta(finish=finish)

    visible, history, spoken, result = exercise(surface, request, monkeypatch, groq(chunks, requests))
    assert len(requests) == 1
    final = requests[0]["messages"][-1]["content"]
    assert SOURCE in final
    assert OMITTED in sources()[1]["document"] and OMITTED not in final
    assert '<source id="2">' not in final
    assert not visible and not history and not spoken
    assert result is not None and result.status != "structurally_validated"


@pytest.mark.parametrize("surface", SURFACES)
def test_only_validated_source_rendering_reaches_normal_sinks(surface, request, monkeypatch):
    payload = packet(1, SOURCE)
    router = groq(lambda: iter((delta(payload), delta(finish="stop"))), [])
    visible, history, spoken, result = exercise(surface, request, monkeypatch, router)
    assert visible and all(SOURCE in text and payload not in text for text in visible)
    assert history[-1]["content"] == visible[-1]
    assert spoken in ([], [visible[-1]])
    assert result.status == "structurally_validated"
    assert result.evidence.context_sha256 and result.evidence.package_sha256

@pytest.mark.parametrize("end", ["timeout", "provider-error", "cancelled"])
def test_pending_background_response_is_never_published(end, load_streamlit_app, monkeypatch):
    import threading

    app, display = load_streamlit_app
    display.button = lambda *a, **k: False
    display.session_state.allow_cloud_speech = True
    spoken = []
    monkeypatch.setattr(app, "ensure_tts_worker", lambda: {"queue": SimpleNamespace(put=spoken.append)})
    pending, release = threading.Event(), threading.Event()
    payload = packet(1, SOURCE)

    def chunks():
        yield delta(payload)
        pending.set()
        assert release.wait(5)
        if end == "timeout":
            raise TimeoutError("private provider fixture")
        if end == "provider-error":
            raise RuntimeError("private provider fixture")
        yield delta(finish="stop")

    job = app.start_generation_job(
        QUESTION, "Use fixture evidence.", preferred=MODEL, router=groq(chunks, []),
        retrieve=lambda *a, **k: (SOURCE, sources()[:1]),
    )
    try:
        assert pending.wait(5)
        app._generation_observer_impl()
        if end == "cancelled":
            assert app.request_generation_cancel(job)
    finally:
        release.set()
        job.thread.join(timeout=5)
    assert not job.thread.is_alive()
    app._generation_observer_impl()
    assert not display.plain_text
    assert not display.session_state.conversation_history
    assert not spoken
    assert display.session_state.current_sources
    assert all("private provider fixture" not in item for item in display.errors)
    assert display.session_state.last_answer_result.status == {
        "timeout": "timeout", "provider-error": "provider_failed", "cancelled": "cancelled",
    }[end]


def test_rejected_qt_response_never_enters_export_or_speech(qapp, monkeypatch, tmp_path):
    import sys

    import nest_asyncio
    from PyQt5.QtWidgets import QTextEdit
    from GUI_chat import ChatFunctionalityMixin
    from GUI_workers import ChatWorker

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(nest_asyncio, "apply", lambda: None)
    import rag_gui

    speech = []
    payload = packet(1, "Synthetic fabricated quotation canary.")

    class Chat(ChatFunctionalityMixin):
        def __init__(self):
            button = SimpleNamespace(setEnabled=lambda enabled: None)
            self.chat_display = QTextEdit()
            self.conversation_history = []
            self.send_btn = self.search_only_btn = self.stop_btn = button
            self.statusBar = SimpleNamespace(showMessage=lambda *a: None)
            self.chat_worker = None
            self.tts_enabled = True

        def set_progress_stage(self, stage):
            pass

        def _play_next_tts_sentence(self):
            speech.extend(self._tts_sentences)

    chat = Chat()
    worker = ChatWorker(
        QUESTION, {"system_message": "Use fixture evidence.", "groq_model": MODEL},
        sources()[:1], chat.conversation_history,
        router=groq(lambda: iter((delta(payload), delta(finish="stop"))), []),
    )
    worker.response_chunk.connect(chat.on_response_chunk)
    worker.response_complete.connect(chat.on_response_complete)
    worker.run()

    for ext in ("txt", "html", "json"):
        path = tmp_path / f"export.{ext}"
        monkeypatch.setattr(rag_gui.QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
        rag_gui.RAGChatGUI.export_chat(chat)
        assert "Synthetic fabricated quotation canary" not in path.read_text(encoding="utf-8")
    assert not speech
    assert chat.conversation_history == []



@pytest.mark.parametrize("payload", [
    '{"excerpts":[{"alias":1,"alias":2,"quote":"valid"}]}',
    '{"excerpts":[{"alias":1.0,"quote":"valid"}]}',
    '{"excerpts":[{"alias":"1","quote":"valid"}]}',
    '{"excerpts":[{"alias":NaN,"quote":"valid"}]}',
    '{"excerpts":[{"alias":1,"quote":"valid","comment":"unexpected"}]}',
    r'{"excerpts":[{"alias":1,"quote":"\ud800"}]}',
    '{"excerpts":[]}',
    '{"excerpts":[{"alias":1,"quote":"  "}]}',
    r'{"excerpts":[{"alias":1,"quote":"\u001b[31m"}]}',
    '{"excerpts":[]}{"excerpts":[]}',
    chr(96) * 3 + 'json\n{"excerpts":[]}\n' + chr(96) * 3,
    json.dumps({"excerpts": [{"alias": 1, "quote": "line\r\n"}, {"alias": 1, "quote": "line\n"}]}),
    pytest.param(packet(1, "x" * 4097), id="quote-limit"),
    pytest.param(packet(1, "x" * 70_000), id="response-limit"),
])
def test_parser_rejects_ambiguous_or_out_of_contract_values(payload):
    from rag_prompting import parse_answer

    with pytest.raises(ValueError, match="Invalid structured answer"):
        parse_answer(payload)


def test_sent_snapshot_is_immutable_and_preserves_compound_identity_and_hashes():
    from hashlib import sha256

    from rag_prompting import (
        build_numbered_sources, parse_answer, prepare_structural_answer, validate_excerpts,
    )

    local = sources()
    local[0].update(file_name="one/shared.txt", meta={"file_path": "one/shared.txt", "page": 2})
    local[1].update(file_name="two/shared.txt", collection_id="another-collection", chunk_id="chunk-1")
    local[1]["document"] = "prelude e\u0301vidence\r\nline  e\u0301vidence\r\nline"
    prepared = prepare_structural_answer("System", QUESTION, build_numbered_sources(local))
    sent_context = prepared.messages()[-1]["content"].split("\n\nQuery: ", 1)[0]
    saved_package = prepared.evidence.package_sha256
    local[0]["meta"]["page"] = 99
    local[1]["document"] = "injected after snapshot"
    prepared.evidence.sources[0]["meta"]["page"] = 13

    assert prepared.evidence.context_sha256 == sha256(sent_context.encode("utf-8")).hexdigest()
    assert saved_package == sha256(prepared.evidence.rows_json.encode("utf-8")).hexdigest()
    assert prepared.evidence.sources[0]["meta"]["page"] == 2
    assert [(row["collection_id"], row["chunk_id"]) for row in prepared.evidence.sources] == [
        ("synthetic", "chunk-1"), ("another-collection", "chunk-1"),
    ]
    effective_quote = validate_excerpts(parse_answer(packet(2, "évidence\nline")), prepared.evidence)
    assert [(record.alias, record.quote) for record in effective_quote] == [(2, "évidence\nline")]
    for modified in ("Évidence\nline", "évidence\n line", "évidence ... line"):
        with pytest.raises(ValueError, match="did not match sent evidence"):
            validate_excerpts(parse_answer(packet(2, modified)), prepared.evidence)


@pytest.mark.parametrize("duplicate", ["alias", "stable-pair"])
def test_evidence_snapshot_rejects_ambiguous_identity(duplicate):
    from rag_prompting import build_evidence_package, build_numbered_sources

    numbered = build_numbered_sources(sources())
    if duplicate == "alias":
        numbered[1]["citation_id"] = numbered[0]["citation_id"]
    else:
        numbered[1]["chunk_id"] = numbered[0]["chunk_id"]
    with pytest.raises(ValueError):
        build_evidence_package(numbered)


@pytest.mark.parametrize("target", ["current", "history"])
def test_mutated_request_is_rejected_before_generation(target):
    from rag_prompting import build_numbered_sources, generate_structural_answer, prepare_structural_answer

    prepared = prepare_structural_answer(
        "System", QUESTION, build_numbered_sources(sources()),
        [{"role": "assistant", "content": "earlier response"}],
    )
    messages = prepared.messages()
    messages[-1 if target == "current" else 1]["content"] += " mutated"
    router = SimpleNamespace(generate=lambda **_: pytest.fail("Mutated request reached provider"))
    result = generate_structural_answer(router, prepared, messages=messages)
    assert result.status == "input_budget" and result.text == ""
    assert result.evidence == prepared.evidence


@pytest.mark.parametrize(("tail", "status"), [
    ("post-stop-error", "provider_failed"),
    ("duplicate-stop", "incomplete"),
    ("post-stop-text", "incomplete"),
    ("usage-only", "structurally_validated"),
    ("invalid-type", "invalid_response"),
    ("tool-delta", "invalid_response"),
    ("oversized", "incomplete"),
])
def test_transport_terminal_semantics_after_complete_json(tail, status):
    from rag_prompting import build_numbered_sources, generate_structural_answer, prepare_structural_answer

    prepared = prepare_structural_answer("System", QUESTION, build_numbered_sources(sources()[:1]))

    def chunks():
        if tail == "oversized":
            yield delta("é" * 32_769)
        yield delta(packet(1, SOURCE))
        yield delta(finish="stop")
        if tail == "post-stop-error":
            raise RuntimeError("private provider failure")
        if tail == "duplicate-stop":
            yield delta(finish="stop")
        if tail == "post-stop-text":
            yield delta("late")
        if tail == "usage-only":
            yield {"choices": [], "usage": {"total_tokens": 12}}
        if tail == "invalid-type":
            yield delta(["extra packet"])
        if tail == "tool-delta":
            yield {"choices": [{"delta": {"tool_calls": [{"id": "unexpected-tool"}]}}]}

    result = generate_structural_answer(
        groq(chunks, []), prepared, provider="groq", preferred=MODEL, strict=True,
    )
    assert result.status == status and bool(result.text) == (status == "structurally_validated")
    assert not result.excerpts or bool(result.text)


@pytest.mark.parametrize(("last_event", "status"), [
    ({"done": True, "done_reason": "stop"}, "structurally_validated"),
    ({"done": True, "done_reason": "length"}, "incomplete"),
    ({"done": True}, "incomplete"),
    ({"done": False, "done_reason": "stop"}, "incomplete"),
    ({"done": "true", "done_reason": "stop"}, "incomplete"),
])
def test_ollama_completion_requires_done_and_stop(last_event, status):
    from generation_models import ollama_router
    from rag_prompting import build_numbered_sources, generate_structural_answer, prepare_structural_answer

    class LocalClient:
        def list(self):
            return {"models": [{"model": "phi-3"}]}

        def show(self, model):
            return {"capabilities": ["completion"], "model_info": {"context_length": 4096}}

        def chat(self, **request):
            assert request["messages"] == prepared.messages()
            yield {"done": False, "done_reason": "", "model": "phi-3",
                   "message": {"content": packet(1, SOURCE)}}
            yield last_event

    router = ollama_router(client=LocalClient(), host="http://localhost:11434")
    prepared = prepare_structural_answer("System", QUESTION, build_numbered_sources(sources()[:1]))
    result = generate_structural_answer(router, prepared, preferred="phi-3", provider="ollama", strict=True)
    assert result.status == status


def test_streamlit_cancel_after_worker_completion_preempts_publication(load_streamlit_app, monkeypatch):
    app, display = load_streamlit_app
    monkeypatch.setattr(app, "ensure_tts_worker", lambda: pytest.fail("Rejected answer spoke"))
    display.session_state.allow_cloud_speech = True
    router = groq(lambda: iter((delta(packet(1, SOURCE)), delta(finish="stop"))), [])
    job = app.start_generation_job(
        QUESTION, "Use fixture evidence.", preferred=MODEL, router=router,
        retrieve=lambda *args, **kwargs: (SOURCE, sources()[:1]),
    )
    job.thread.join(timeout=5)
    assert not job.thread.is_alive()
    assert app.request_generation_cancel(job)
    app._generation_observer_impl()
    assert display.session_state.last_answer_result.status == "cancelled"
    assert display.session_state.current_sources[0]["document"] == SOURCE
    assert not display.plain_text and not display.session_state.conversation_history


@pytest.mark.parametrize("accepted", [False, True])
def test_streamlit_export_preserves_the_import_contract(load_streamlit_app, monkeypatch, accepted):
    from conversation_import import sanitize_conversation_import

    app, display = load_streamlit_app
    quote_canary = "Fabricated export canary."
    quote = SOURCE if accepted else quote_canary
    router = groq(lambda: iter((delta(packet(1, quote)), delta(finish="stop"))), [])
    app._initialize_session()
    app.chat_with_model(
        QUESTION, "Use fixture evidence.", preferred=MODEL, router=router,
        retrieve=lambda *args, **kwargs: (SOURCE, sources()[:1]),
    )
    display.setup_main("ollama")
    display.button = lambda label, **kwargs: label == "Export Conversation"
    exported = []
    display.download_button = lambda label, *, data, file_name: exported.append(data)
    monkeypatch.setattr(app, "_generation_settings", lambda **kwargs: ("ollama", router, MODEL, True))
    monkeypatch.setattr(app, "generate_word_cloud", lambda _: SimpleNamespace(to_array=lambda: []))
    plotly = ModuleType("plotly")
    plotly.express = ModuleType("plotly.express")
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.express", plotly.express)
    app.main()
    assert len(exported) == 1 and quote_canary not in exported[0]
    result = json.loads(exported[0])
    imported, _ = sanitize_conversation_import(exported[0].encode("utf-8"))
    if accepted:
        assert SOURCE in imported["history"][-1]["content"]
    else:
        assert imported["history"] == []
    assert result["sources"][0]["document"] == SOURCE


def test_catalogue_timeout_is_typed_with_evidence_and_without_provider_text():
    from rag_prompting import build_numbered_sources, generate_structural_answer, prepare_structural_answer

    def list_models():
        raise TimeoutError("synthetic private provider diagnostic")

    router = GroqRouter(client=SimpleNamespace(models=SimpleNamespace(list=list_models)))
    prepared = prepare_structural_answer("System", QUESTION, build_numbered_sources(sources()[:1]))
    result = generate_structural_answer(router, prepared, provider="groq")
    assert result.status == "timeout" and result.text == ""
    assert result.evidence == prepared.evidence
    assert "private provider" not in result.message
