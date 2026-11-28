"""Maintained Qt LM Studio contract, using no model or corpus files."""

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from generation_models import GenerationError
from rag_prompting import AnswerStatus
from test_generation_qt import qapp


MODEL = "fixture-ornith"
MARKER = "<|assistant|>\n"
SYSTEM = {"role": "system", "content": "Use the evidence."}
QUERY = {"role": "user", "content": "Atlas: π blue.\nWhich color?"}
MESSAGES = [SYSTEM, QUERY]


def render(messages):
    return "<s>\n" + "".join(m["content"] + "\n" for m in messages) + MARKER


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def deployment():
    from lmstudio_generation import Deployment, RENDER_PROBE

    return Deployment(
        model_id=MODEL, model_key="fixture/model", model_path="fixture/model.gguf",
        size_bytes=456, instance_reference="fixture-instance-reference",
        context_tokens=64, output_tokens=16, generation_marker=MARKER,
        render_probe_sha256=digest(render(RENDER_PROBE)),
        framing_min=0, framing_max=2, evidence_sha256="e" * 64,
    )


class Session:
    def __init__(self, deployment):
        self.identity = deployment.identity()
        self.metadata = {**self.identity, "effectiveContext": 64}
        self.requests = []
        self.used = []
        self.count = 5
        self.render_override = None
        self.render_digest_override = None
        self.terminal = {}
        self.failure = None
        self.during = None
        self.released = False
        self.busy = False

    @contextmanager
    def open(self, deployment, cancelled):
        if self.busy:
            raise GenerationError("resource_busy")
        self.used.append("open")
        try:
            yield self
        finally:
            self.released = True

    def request(self, command, body, cancelled):
        self.requests.append(command)
        if command == "inspect":
            return self.metadata
        if command == "prepare":
            output = self.render_override or render(body["messages"])
            self.used.append(output)
            return {
                "rendered": output, "promptSha256": self.render_digest_override or digest(output),
                "nativeTokens": self.count, "effectiveContext": self.metadata.get("effectiveContext"),
            }
        assert command == "predict"
        if self.during:
            self.during()
        if self.failure:
            raise self.failure
        assert body["promptSha256"] == digest(self.used[-1])
        return {
            "text": "fixture answer", "stopReason": "eosFound",
            "promptTokens": self.count + 1, "outputTokens": 2,
            "maxTokens": body["maxTokens"], "overflow": "stopAtLimit",
            "effectiveContext": 64, "identity": self.identity,
            **self.terminal,
        }


def generate(deployment, session, **options):
    from lmstudio_generation import LMStudioRouter

    router = LMStudioRouter(deployment, session_factory=session.open)
    return router.generate(**{
        "preferred": MODEL, "messages": MESSAGES, "mode": "local-chat",
        "max_output_tokens": 12, "text_only": True, "answer_buffer_bytes": 128,
        **options,
    })


def test_native_count_and_full_output_are_reserved_at_limit(deployment):
    session = Session(deployment)
    session.count = 50
    result = generate(deployment, session)
    assert result.provider == "lmstudio"
    assert result.finish_reason == "stop"
    assert result.admission.input_count.tokens == 52
    assert result.admission.output_tokens == 12
    assert result.admission.context_tokens == 64
    assert session.requests == ["inspect", "prepare", "predict"]
    assert session.released


def test_admission_drops_whole_history_turn_only(deployment):
    session = Session(deployment)
    original_request = session.request

    def request(command, body, cancelled):
        if command == "prepare":
            session.count = 51 if len(body["messages"]) > 2 else 50
        return original_request(command, body, cancelled)

    session.request = request
    history = [{"role": "user", "content": "prior query"}, {"role": "assistant", "content": "prior answer"}]
    outcome = generate(deployment, session, messages=[SYSTEM, *history, QUERY])
    assert outcome.admission.history_turns_removed == 1
    assert session.used[-1] == render([SYSTEM, QUERY])
    assert session.requests == ["inspect", "prepare", "prepare", "predict"]


def test_lower_operator_context_limit_keeps_loaded_capacity_binding(deployment):
    deployment = replace(deployment, context_ceiling=62)
    session = Session(deployment)
    session.count = 49
    with pytest.raises(GenerationError, match="fit"):
        generate(deployment, session)
    assert "predict" not in session.requests


@pytest.mark.parametrize("context", [None, True, 0, 63, 65])
def test_capacity_must_match_verified_loaded_value(deployment, context):
    session = Session(deployment)
    session.metadata["effectiveContext"] = context
    with pytest.raises(GenerationError):
        generate(deployment, session)
    assert "predict" not in session.requests


@pytest.mark.parametrize("field", ["identifier", "modelKey", "instanceReference", "sizeBytes", "path", "deviceIdentifier"])
def test_loaded_identity_mismatch_is_terminal_before_dispatch(deployment, field):
    session = Session(deployment)
    session.metadata[field] = "different"
    with pytest.raises(GenerationError):
        generate(deployment, session)
    assert "predict" not in session.requests


def test_strict_model_pin_and_cloud_request_fail_without_opening_sdk(deployment):
    session = Session(deployment)
    for options in ({"preferred": None}, {"preferred": "other-model"}, {"provider": "groq"},
                    {"mode": "cloud"}, {"tools": []}, {"preset": "unsafe"}):
        with pytest.raises(GenerationError):
            generate(deployment, session, **options)
    assert not session.used


@pytest.mark.parametrize("change", [
    {"render_override": "<s>\nUse the evidence.\nAtlas: π blue.\nWhich color?\n"},
    {"render_override": "<s>\nAtlas: pi blue.\n" + MARKER},
    {"render_digest_override": "bad"},
    {"count": 51},
])
def test_prompt_integrity_and_reservation_refuse_dispatch(deployment, change):
    session = Session(deployment)
    vars(session).update(change)
    with pytest.raises(GenerationError, match="context|fit"):
        generate(deployment, session)
    assert "predict" not in session.requests


def test_missing_framing_evidence_refuses_before_open(deployment):
    session = Session(deployment)
    with pytest.raises(GenerationError):
        generate(replace(deployment, evidence_sha256=""), session)
    assert not session.used


@pytest.mark.parametrize("terminal", [
    {"identity": {}}, {"effectiveContext": 63}, {"maxTokens": 11},
    {"overflow": "truncateMiddle"}, {"stopReason": "contextLengthReached"},
    {"stopReason": "maxPredictedTokensReached"}, {"stopReason": "toolCalls"},
    {"stopReason": "failed"}, {"promptTokens": None}, {"promptTokens": 8},
    {"promptTokens": 4}, {"outputTokens": 13}, {"outputTokens": None},
    {"truncated": True}, {"text": "x" * 129},
])
def test_no_output_on_unverified_or_incomplete_terminal_receipt(deployment, terminal):
    session = Session(deployment)
    session.terminal = terminal
    with pytest.raises(GenerationError):
        generate(deployment, session)
    assert session.requests[-1] == "predict"
    assert session.released


def test_cancellation_before_or_during_prediction(deployment):
    cancelled = threading.Event()
    session = Session(deployment)
    cancelled.set()
    with pytest.raises(GenerationError, match="[Cc]ancel"):
        generate(deployment, session, cancelled=cancelled)
    assert not session.requests
    cancelled.clear()
    session.during = cancelled.set
    with pytest.raises(GenerationError, match="[Cc]ancel"):
        generate(deployment, session, cancelled=cancelled)
    assert session.released


@pytest.mark.parametrize("busy", [True, False])
def test_failure_or_slot_contention_uses_one_local_attempt(deployment, busy):
    session = Session(deployment)
    session.busy = busy
    session.failure = RuntimeError("sensitive fixture details")
    with pytest.raises(GenerationError) as error:
        generate(deployment, session)
    assert "sensitive" not in str(error.value)
    assert session.requests.count("predict") == (0 if busy else 1)
    assert session.released is not busy


@pytest.mark.parametrize("answer", ["valid", "translated", "off", "search"])
def test_qt_uses_structural_validator_and_never_falls_back(qapp, request, monkeypatch, answer):
    import generation_models
    from GUI_context import rewrite_input_and_generate_synonyms
    from GUI_workers import ChatWorker

    def cloud_or_other(*args, **kwargs):
        pytest.fail("selected LM Studio route reached another provider")

    monkeypatch.setattr(generation_models, "groq_router", cloud_or_other)
    monkeypatch.setattr(generation_models, "ollama_router", cloud_or_other)
    settings = {"generation_provider": "lmstudio", "lmstudio_model": MODEL,
                "generation_router": SimpleNamespace(generate=cloud_or_other),
                "system_message": SYSTEM["content"],
                "generation_max_output_tokens": 12, "just_search": answer == "search"}
    session = None
    if answer in {"valid", "translated"}:
        from lmstudio_generation import LMStudioRouter

        deployment = request.getfixturevalue("deployment")
        session = Session(deployment)
        session.terminal["text"] = json.dumps({"excerpts": [
            {"alias": 1, "quote": "Atlas: π blue." if answer == "valid" else "Atlas: pi blue."}
        ]})
        settings["generation_router_lmstudio"] = LMStudioRouter(deployment, session_factory=session.open)
    assert rewrite_input_and_generate_synonyms(settings, "original") == ("original", {})
    worker = ChatWorker("Color on Atlas?", settings, [
        {"collection_id": "test", "chunk_id": "atlas", "file_name": "fixture.txt",
         "document": "Atlas: π blue."},
    ], [])
    results, sink = [], []
    worker.generation_result.connect(results.append)
    worker.response_chunk.connect(sink.append)
    worker.run()
    if answer == "search":
        assert not results and not sink
        return
    expected = {"valid": AnswerStatus.STRUCTURALLY_VALIDATED,
                "translated": AnswerStatus.INVALID_RESPONSE,
                "off": AnswerStatus.PROVIDER_FAILED}[answer]
    assert results[0].status == expected, session.requests if session else None
    assert results[0].provider == "lmstudio"
    assert bool(sink) is (answer == "valid")


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_smoke_fixtures_define_a_structural_and_relevance_boundary(qapp, deployment, case_index):
    from GUI_workers import ChatWorker
    from lmstudio_generation import LMStudioRouter

    fixtures = json.loads(Path(__file__).with_name("lmstudio_smoke_fixtures.json").read_text(encoding="utf-8"))
    row = fixtures["cases"][case_index]
    quote = row["expected_quote"] or row["document"]
    session = Session(deployment)
    session.terminal["text"] = json.dumps({"excerpts": [{"alias": 1, "quote": quote}]})
    history, results, sink = [], [], []
    settings = {"generation_provider": "lmstudio", "lmstudio_model": MODEL,
                "system_message": fixtures["system"], "generation_max_output_tokens": 12}
    worker = ChatWorker(row["query"], settings, [
        {"collection_id": "synthetic-lmstudio-smoke", "chunk_id": row["chunk"],
         "file_name": row["id"] + ".txt", "document": row["document"]},
    ], history, router=LMStudioRouter(deployment, session_factory=session.open))
    worker.generation_result.connect(results.append)
    worker.response_chunk.connect(sink.append)
    worker.run()
    assert results[0].status is AnswerStatus.STRUCTURALLY_VALIDATED
    assert results[0].evidence.context_sha256
    assert results[0].excerpts[0].quote == quote
    assert len(sink) == 1 and len(history) == 2
    # A verbatim but irrelevant excerpt would fail the later live smoke's
    # relevance criterion. The structural validator accepts its exact span.
    assert (quote == row["expected_quote"]) is (row["id"] != "insufficient")
