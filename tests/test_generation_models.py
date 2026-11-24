from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import groq
import httpx
import pytest

import generation_models as gm


def _chunk(text: str, model: str = "") -> SimpleNamespace:
    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(delta=delta, message=SimpleNamespace(content=None))
    return SimpleNamespace(choices=[choice], model=model)


class FakeGroq:
    def __init__(self, models: list[dict[str, object]]) -> None:
        self.events: list[tuple[str, str]] = []
        self.models_rows = models
        self.calls: list[dict[str, object]] = []
        self.models = SimpleNamespace(list=self._list)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._missing_once = True

    def _list(self):
        self.events.append(("list", ""))
        return SimpleNamespace(
            data=[SimpleNamespace(**row) for row in self.models_rows]
        )

    def _create(self, **kwargs):
        self.events.append(("create", str(kwargs["model"])))
        self.calls.append(kwargs)
        if kwargs["model"] == "llama-3.3-70b-versatile" and self._missing_once:
            self._missing_once = False
            response = httpx.Response(
                404,
                request=httpx.Request("POST", "https://example.invalid"),
            )
            raise groq.NotFoundError(
                "model not found", response=response, body={"error": {"code": "model_not_found"}}
            )
        return iter([_chunk("Answer [doc-1]", str(kwargs["model"]))])


def _groq_rows(include_replacement: bool = False) -> list[dict[str, object]]:
    rows = [{"id": "llama-3.3-70b-versatile", "active": True}]
    if include_replacement:
        rows.append({"id": "openai/gpt-oss-120b", "active": True})
    return rows


def test_catalogue_is_ordered_and_exposes_profiles() -> None:
    client = FakeGroq(_groq_rows(True))
    router = gm.groq_router(client=client)

    profiles = router.catalogue()

    assert [profile.id for profile in profiles] == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
    assert "stream" in profiles[0].capabilities
    assert profiles[0].provider == "groq"


def test_typed_404_refreshes_same_client_and_falls_back_without_changing_request() -> None:
    client = FakeGroq(_groq_rows(False))
    router = gm.groq_router(client=client)
    router.catalogue()
    client.events.clear()
    client.models_rows = _groq_rows(True)
    messages = [
        {"role": "system", "content": "Answer with evidence."},
        {"role": "user", "content": "What happened in [doc-1]?"},
    ]

    result = router.generate(
        preferred="llama-3.3-70b-versatile",
        messages=messages,
        stream=True,
        max_output_tokens=512,
    )

    assert [model for kind, model in client.events if kind == "create"] == [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ]
    assert [kind for kind, _ in client.events] == ["create", "list", "create"]
    assert client.calls[0]["messages"] == client.calls[1]["messages"] == messages
    assert client.calls[0]["service_tier"] == client.calls[1]["service_tier"] == "on_demand"
    assert result.text == "Answer [doc-1]"
    assert result.preferred_model == "llama-3.3-70b-versatile"
    assert result.actual_model == "openai/gpt-oss-120b"
    assert "openai/gpt-oss-120b" in result.reason


def test_strict_pin_does_not_retry() -> None:
    client = FakeGroq(_groq_rows(False))
    router = gm.groq_router(client=client)
    router.catalogue()
    client.models_rows = _groq_rows(True)

    with pytest.raises(gm.GenerationError) as caught:
        router.generate(preferred="llama-3.3-70b-versatile", messages=[], strict=True)

    assert caught.value.code == "model_unavailable"
    assert len(client.calls) == 1
    assert not any(kind == "list" for kind, _ in client.events[1:])


def test_auth_failure_is_terminal() -> None:
    client = FakeGroq(_groq_rows(True))
    router = gm.groq_router(client=client)
    router.catalogue()

    class AuthError(Exception):
        status_code = 401
        body = {"error": {"code": "invalid_api_key"}}

    def fail(**kwargs):
        client.calls.append(kwargs)
        raise AuthError("secret should not leak")

    client.chat.completions.create = fail
    with pytest.raises(gm.GenerationError) as caught:
        router.generate(preferred="llama-3.3-70b-versatile", messages=[])

    assert caught.value.code == "authentication"
    assert len(client.calls) == 1
    assert "secret" not in str(caught.value).lower()


@pytest.mark.parametrize("tool_delta", [False, True])
def test_no_retry_after_first_output_delta(tool_delta):
    client = FakeGroq(_groq_rows(True))
    router = gm.groq_router(client=client)
    router.catalogue()

    def stream_then_fail(**kwargs):
        client.calls.append(kwargs)

        def _stream():
            if tool_delta:
                delta = SimpleNamespace(content=None, tool_calls=[{"index": 0}])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
            else:
                yield _chunk("partial")
            response = httpx.Response(
                404,
                request=httpx.Request("POST", "https://example.invalid"),
            )
            raise groq.NotFoundError(
                "model not found",
                response=response,
                body={"error": {"code": "model_not_found"}},
            )

        return _stream()

    client.chat.completions.create = stream_then_fail
    received: list[str] = []

    with pytest.raises(gm.GenerationError) as caught:
        router.generate(
            preferred="llama-3.3-70b-versatile",
            messages=[],
            on_chunk=received.append,
        )

    assert caught.value.code == "stream_error"
    assert received == ([] if tool_delta else ["partial"])
    assert len(client.calls) == 1


def test_unrelated_groq_404_is_terminal() -> None:
    client = FakeGroq(_groq_rows(True))
    router = gm.groq_router(client=client)
    router.catalogue()

    def fail(**kwargs):
        client.calls.append(kwargs)
        response = httpx.Response(
            404,
            request=httpx.Request("POST", "https://example.invalid"),
        )
        raise groq.NotFoundError(
            "not found",
            response=response,
            body={"error": {"code": "route_not_found"}},
        )

    client.chat.completions.create = fail
    with pytest.raises(gm.GenerationError) as caught:
        router.generate(preferred="llama-3.3-70b-versatile", messages=[])

    assert caught.value.code == "provider_error"
    assert len(client.calls) == 1


def test_three_typed_preoutput_failures_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path(__file__).parents[1] / "config" / "generation-models.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["providers"]["groq"]["compact/test-1"] = dict(
        config["providers"]["groq"]["llama-3.1-8b-instant"]
    )
    config["providers"]["groq"]["compact/test-2"] = dict(
        config["providers"]["groq"]["llama-3.1-8b-instant"]
    )
    override = tmp_path / "profiles.json"
    override.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("EASY_RAG_GENERATION_PROFILES", str(override))
    rows = [
        {"id": "llama-3.1-8b-instant", "active": True},
        {"id": "compact/test-1", "active": True},
        {"id": "compact/test-2", "active": True},
    ]
    client = FakeGroq(rows)
    router = gm.groq_router(client=client)
    router.catalogue()

    def fail(**kwargs):
        client.calls.append(kwargs)
        response = httpx.Response(
            404,
            request=httpx.Request("POST", "https://example.invalid"),
        )
        raise groq.NotFoundError(
            "model not found",
            response=response,
            body={"error": {"code": "model_not_found"}},
        )

    client.chat.completions.create = fail
    with pytest.raises(gm.GenerationError) as caught:
        router.generate(preferred="llama-3.1-8b-instant", messages=[])

    assert caught.value.code == "model_unavailable"
    assert len(client.calls) == 3


def test_saved_known_preference_can_be_refreshed_to_replacement() -> None:
    client = FakeGroq([{"id": "openai/gpt-oss-120b", "active": True}])
    router = gm.groq_router(client=client)

    def create(**kwargs):
        client.calls.append(kwargs)
        if kwargs["model"] == "llama-3.3-70b-versatile":
            response = httpx.Response(
                404,
                request=httpx.Request("POST", "https://example.invalid"),
            )
            raise groq.NotFoundError(
                "model not found",
                response=response,
                body={"error": {"code": "model_not_found"}},
            )
        return iter([_chunk("replacement", kwargs["model"])])

    client.chat.completions.create = create
    result = router.generate(preferred="llama-3.3-70b-versatile", messages=[])

    assert [call["model"] for call in client.calls] == [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ]
    assert result.preferred_model == "llama-3.3-70b-versatile"
    assert result.actual_model == "openai/gpt-oss-120b"


def test_ollama_remote_manifest_is_visible_but_not_dispatchable() -> None:
    class RemoteOllama:
        def list(self):
            return {"models": [{"name": "llama3.2:3b", "remote_model": True}]}

        def show(self, model):
            return {"capabilities": ["completion"], "remote_host": "cloud.example"}

    router = gm.ollama_router(client=RemoteOllama(), host="http://localhost:11434")
    profiles = router.catalogue()

    assert profiles[0].remote is True
    with pytest.raises(gm.GenerationError) as caught:
        router.generate(preferred="llama3.2:3b", messages=[])
    assert caught.value.code == "no_candidate"


def test_ollama_show_failure_is_unverified_and_not_dispatchable() -> None:
    class IncompleteOllama:
        def list(self):
            return {"models": [{"name": "phi-3"}]}

        def show(self, model):
            raise RuntimeError("show unavailable")

        def chat(self, **kwargs):
            pytest.fail("unverified model was dispatched")

    router = gm.ollama_router(client=IncompleteOllama(), host="http://localhost:11434")
    profiles = router.catalogue()

    assert profiles[0].profiled is False
    with pytest.raises(gm.GenerationError) as caught:
        router.generate(preferred="phi-3", messages=[])
    assert caught.value.code == "no_candidate"


def test_ollama_checks_effective_context_before_dispatch():
    class LocalOllama:
        def list(self):
            return {"models": [{"model": "phi-3"}]}

        def show(self, model):
            return {"capabilities": ["completion"], "model_info": {"context_length": 4096}}

        def chat(self, **kwargs):
            pytest.fail("Oversized request reached Ollama")

    router = gm.ollama_router(client=LocalOllama(), host="http://localhost:11434")
    for effective_context in (0, 256):
        with pytest.raises(gm.GenerationError) as caught:
            router.generate(
                preferred="phi-3",
                messages=[{"role": "user", "content": "evidence " * 40}],
                max_output_tokens=32,
                options={"num_ctx": effective_context},
            )
        assert caught.value.code == "no_candidate"


def test_policy_denies_capacity_and_strict_json_streaming() -> None:
    client = FakeGroq(_groq_rows(True))
    router = gm.groq_router(client=client)
    router.catalogue()

    with pytest.raises(gm.GenerationError) as capacity:
        router.generate(messages=[{"role": "user", "content": "x"}], max_output_tokens=999999)
    assert capacity.value.code == "no_candidate"

    with pytest.raises(gm.GenerationError) as schema:
        router.generate(
            messages=[],
            stream=True,
            response_format={"type": "json_schema", "json_schema": {"strict": True}},
        )
    assert schema.value.code == "no_candidate"


def test_groq_does_not_switch_a_requested_service_tier():
    client = FakeGroq(_groq_rows(True))
    router = gm.groq_router(client=client)

    with pytest.raises(gm.GenerationError):
        router.generate(
            preferred="llama-3.3-70b-versatile", messages=[], service_tier="flex"
        )

    assert client.calls == []


def test_cancellation_is_checked_before_discovery() -> None:
    client = FakeGroq(_groq_rows(True))
    router = gm.groq_router(client=client)

    with pytest.raises(gm.GenerationError) as caught:
        router.generate(messages=[], cancelled=lambda: True)

    assert caught.value.code == "cancelled"
    assert client.events == []


def test_catalogue_ttl_is_per_router_and_refreshes_at_expiry() -> None:
    now = [100.0]
    left = FakeGroq(_groq_rows(True))
    right = FakeGroq(_groq_rows(True))
    left_router = gm.groq_router(client=left, clock=lambda: now[0])
    right_router = gm.groq_router(client=right, clock=lambda: now[0])

    left_router.catalogue()
    left_router.catalogue()
    right_router.catalogue()
    assert [kind for kind, _ in left.events] == ["list"]
    assert [kind for kind, _ in right.events] == ["list"]

    now[0] += 301
    left_router.catalogue()
    assert [kind for kind, _ in left.events] == ["list", "list"]


def test_unknown_models_are_visible_but_not_automatic_candidates() -> None:
    client = FakeGroq([{"id": "new-model", "active": True}])
    router = gm.groq_router(client=client)

    profiles = router.catalogue()

    assert profiles[0].id == "new-model"
    assert profiles[0].profiled is False
    with pytest.raises(gm.GenerationError) as caught:
        router.generate(messages=[])
    assert caught.value.code == "no_candidate"
