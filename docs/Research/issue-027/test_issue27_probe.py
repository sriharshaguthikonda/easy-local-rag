"""Verification probes for the unmodified issue #27 baseline.

These assertions describe observed behavior at 656ae2a905cb, including defects.
Run against the pinned source with Python network calls disabled.
"""

from pathlib import Path
import os
import sys
import threading
from types import SimpleNamespace

import pytest

assert os.environ.get("ISSUE27_SOCKET_DENIAL") == "active"
sys.path.insert(0, str(Path.cwd() / "tests"))

from PyQt5.QtWidgets import QTextEdit
from GUI_chat import ChatFunctionalityMixin
from GUI_workers import ChatWorker
from generation_models import GroqRouter
from test_generation_qt import qapp
from test_generation_streamlit import load_streamlit_app

MODEL = "llama-3.3-70b-versatile"
RED = "The synthetic Atlas status lamp is red."
GREEN = "The synthetic Atlas status lamp is green."
CABLE = "The synthetic Atlas includes a cable."
OMITTED = "The omitted synthetic Atlas record says the lamp is violet."
QUESTION = "What color is the synthetic Atlas status lamp?"

CASES = [
    ("supported", (RED,), "The lamp is red [1].", "stop", True),
    ("invented", (RED,), "The lamp is red [99].", "stop", False),
    ("unsupported_with_valid_id", (RED,), "The Atlas comes with a cable [1].", "stop", True),
    ("fabricated_quote", (RED,), '"The Atlas is glowing red" [1].', "stop", True),
    ("contradicted", (RED,), "The lamp is blue [1].", "stop", True),
    ("conflict_suppressed", (RED, GREEN), "All sources agree the lamp is red [1].", "stop", True),
    ("nonanswering_low_score", (CABLE,), "The lamp is definitely blue.", "stop", True),
    ("excluded_chunk", (RED, OMITTED), "The omitted record says violet [2].", "stop", False),
    ("excluded_chunk_noncanonical", (RED, OMITTED), "The omitted record says violet [2, 99].", "stop", True),
    ("noncanonical_marker", (RED,), "The lamp is red [99, 100].", "stop", True),
    ("malformed_contract", (RED,), '{"answer": "red [1]"', "stop", True),
    ("empty_provider_content", (RED,), "", "stop", True),
    ("length_finish", (RED,), "The lamp is red [1] because", "length", True),
]


def make_sources(documents):
    return [
        {
            "collection_id": "synthetic-collection",
            "chunk_id": "synthetic-chunk-%s" % index,
            "file_name": "synthetic-missing-file-%s.txt" % index,
            "meta": {"file_name": "synthetic-missing-file-%s.txt" % index},
            "document": document,
            "final_score": 0.0,
            "score": 0.0,
        }
        for index, document in enumerate(documents, 1)
    ]


def make_router(text, finish, requests):
    def create(**request):
        requests.append(request)
        assert request["stream"] is True
        return iter((
            {"model": MODEL, "choices": [{"delta": {"content": text}}]},
            {"model": MODEL, "choices": [{"delta": {}, "finish_reason": finish}]},
        ))

    return GroqRouter(client=SimpleNamespace(
        models=SimpleNamespace(list=lambda: [{"id": MODEL}]),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))


class QtDisplay(ChatFunctionalityMixin):
    def __init__(self):
        self.chat_display = QTextEdit()

    def set_progress_stage(self, stage):
        pass


@pytest.mark.parametrize(
    "case,documents,text,finish,streamlit_accepts",
    [pytest.param(*case, id=case[0]) for case in CASES],
)
@pytest.mark.parametrize("surface", ["qt", "streamlit_sync", "streamlit_background"])
def test_response_acceptance(case, documents, text, finish, streamlit_accepts, surface, request):
    all_candidates = make_sources(documents)
    sources = all_candidates[:1] if case.startswith("excluded_chunk") else all_candidates
    requests = []
    router = make_router(text, finish, requests)
    if surface == "qt":
        request.getfixturevalue("qapp")
        display = QtDisplay()
        history, complete, statuses, errors = [], [], [], []
        worker = ChatWorker(
            QUESTION,
            {"system_message": "Answer from sources.", "generation_provider": "groq",
             "groq_model": MODEL, "strict_generation_model": True},
            sources, history, router=router,
        )
        worker.response_chunk.connect(display.on_response_chunk)
        worker.response_complete.connect(complete.append)
        worker.status_update.connect(statuses.append)
        worker.error_occurred.connect(errors.append)
        worker.run()
        assert complete == [text]
        assert display.chat_display.toPlainText() == text
        assert history[-1] == {"role": "assistant", "content": text}
        assert statuses[-1] == "Answered with " + MODEL
        assert errors == []
    else:
        app, display = request.getfixturevalue("load_streamlit_app")

        def retrieve(question, **options):
            return "\n\n".join(row["document"] for row in sources), sources

        settings = {"preferred": MODEL, "router": router, "retrieve": retrieve, "strict": True}
        if surface == "streamlit_sync":
            _, answer, _ = app.chat_with_model(QUESTION, "Answer from sources.", **settings)
            assert answer == (text if streamlit_accepts else None)
            if text:
                assert text + "▌" in display.plain_text  # Raw text appeared before validation.
        else:
            job = app.start_generation_job(QUESTION, "Answer from sources.", **settings)
            job.thread.join(timeout=5)
            assert not job.thread.is_alive()
            app._generation_observer_impl()
        if streamlit_accepts:
            assert display.plain_text[-1] == text
            assert display.session_state.conversation_history[-1]["content"] == text
            assert display.errors == []
            if case in {"supported", "unsupported_with_valid_id", "fabricated_quote",
                        "contradicted", "conflict_suppressed", "malformed_contract", "length_finish"}:
                assert display.session_state.current_citations == [1]
            if case in {"nonanswering_low_score", "noncanonical_marker", "excluded_chunk_noncanonical"}:
                assert display.session_state.last_generation_warning == "No citations returned in response."
        else:
            assert display.session_state.conversation_history == []
            assert display.errors == ["The answer cited an unknown source."]
            assert text not in display.plain_text
    assert len(requests) == 1
    assert "response_format" not in requests[0]
    current_turn = requests[0]["messages"][-1]["content"]
    assert all(row["document"] in current_turn for row in sources)
    if case.startswith("excluded_chunk"):
        assert OMITTED in all_candidates[1]["document"]
        assert OMITTED not in current_turn
        assert '<source id="2">' not in current_turn


@pytest.mark.parametrize("surface", ["qt", "streamlit"])
@pytest.mark.parametrize("kind", ["empty", "missing_document", "overflow"])
def test_input_guards(surface, kind, request):
    requests = []
    router = make_router("Unexpected answer [1].", "stop", requests)
    if kind == "empty":
        sources = []
    elif kind == "missing_document":
        sources = make_sources(("",))
    else:
        sources = make_sources(("evidence " * 40_000,))
    if surface == "qt":
        request.getfixturevalue("qapp")
        history, chunks, errors = [], [], []
        worker = ChatWorker(
            QUESTION,
            {"system_message": "Answer from sources.", "generation_provider": "groq",
             "groq_model": MODEL, "strict_generation_model": True},
            sources, history, router=router,
        )
        worker.response_chunk.connect(chunks.append)
        worker.error_occurred.connect(errors.append)
        worker.run()
        assert errors and not chunks and not history
    else:
        app, display = request.getfixturevalue("load_streamlit_app")
        _, answer, _ = app.chat_with_model(
            QUESTION, "Answer from sources.", preferred=MODEL, strict=True, router=router,
            retrieve=lambda question, **kwargs: ("\n\n".join(row["document"] for row in sources), sources),
        )
        assert answer is None
        assert display.errors
        assert display.session_state.conversation_history == []
    assert requests == []


def test_background_can_show_invalid_citation_before_rejecting_it(load_streamlit_app):
    app, display = load_streamlit_app
    display.button = lambda *args, **kwargs: False
    first_chunk_seen, allow_finish = threading.Event(), threading.Event()
    text = "The lamp is red [99]."

    def create(**options):
        def chunks():
            yield {"model": MODEL, "choices": [{"delta": {"content": text}}]}
            first_chunk_seen.set()
            if not allow_finish.wait(5):
                raise TimeoutError("audit fixture did not release stream")
            yield {"model": MODEL, "choices": [{"delta": {}, "finish_reason": "stop"}]}

        return chunks()

    router = GroqRouter(client=SimpleNamespace(
        models=SimpleNamespace(list=lambda: [{"id": MODEL}]),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    job = app.start_generation_job(
        QUESTION, "Answer from sources.", preferred=MODEL, router=router,
        retrieve=lambda question, **kwargs: (RED, make_sources((RED,))),
    )
    try:
        assert first_chunk_seen.wait(5)
        app._generation_observer_impl()
        assert display.plain_text[-1] == text + "▌"
        assert display.session_state.conversation_history == []
    finally:
        allow_finish.set()
        job.thread.join(5)
    assert not job.thread.is_alive()
    app._generation_observer_impl()
    assert display.errors == ["The answer cited an unknown source."]
    assert display.session_state.conversation_history == []
