"""Synthetic regressions for the maintained Chroma answer boundary."""

import json
from types import SimpleNamespace

import pytest

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