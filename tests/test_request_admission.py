"""Network-denied regressions for the maintained request-admission boundary."""

import copy
from types import SimpleNamespace

import pytest

import generation_models as gm
from test_generation_streamlit import load_streamlit_app
from test_structural_answers import MODEL, QUESTION, SOURCE, delta, packet, sources

FALLBACK = "openai/gpt-oss-120b"


def client(rows, handler=None):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if handler:
            return handler(kwargs)
        return iter([delta("accepted"), delta(finish="stop")])

    router = gm.GroqRouter(client=SimpleNamespace(
        models=SimpleNamespace(list=lambda: rows),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    return router, calls


def measured(router, monkeypatch, *, counting, contexts, seen=None):
    """Substitute deployment metadata and verified tokenizer I/O, not admission."""
    def measure(profile, messages, options):
        if seen is not None:
            seen.append((profile.id, copy.deepcopy(messages)))
        from token_budget import TokenCount
        return TokenCount(counting(profile.id, messages), "exact", "synthetic chat template"), contexts[profile.id], "synthetic deployment"

    monkeypatch.setattr(router, "_admission_inputs", measure, raising=False)


@pytest.mark.parametrize("body", [
    "呼吸器内科" * 400,
    "e\u0301 Å 🫁 日本語",
    "N-acetylneuraminic acid and phosphatidylinositol-3-kinase",
    "def f(xs):\n    return [x ** 2 for x in xs]\n" * 200,
], ids=["cjk", "unicode", "biomedical", "code"])
def test_unknown_tokenizer_is_a_capacity_refusal(body):
    router, calls = client([{"id": MODEL, "context_window": 131072}])
    with pytest.raises(gm.GenerationError) as caught:
        router.generate(MODEL, [{"role": "user", "content": body}], max_output_tokens=64)
    assert caught.value.code == "input_budget"
    assert not calls


def test_local_context_unknown_even_with_configured_profile():
    class Local:
        def list(self):
            return {"models": [{"model": "phi-3"}]}

        def show(self, model):
            return {"capabilities": ["completion"]}

        def chat(self, **kwargs):
            pytest.fail("unverified local context reached generation")

    router = gm.OllamaRouter(client=Local())
    with pytest.raises(gm.GenerationError) as caught:
        router.generate("phi-3", [{"role": "user", "content": "question"}], max_output_tokens=64)
    assert caught.value.code == "input_budget"


def test_exact_boundary_and_reserved_output_use_candidate_count(monkeypatch):
    router, calls = client([{"id": MODEL, "context_window": 820}])
    messages = [{"role": "system", "content": "instructions"},
                {"role": "user", "content": "糖 " * 1000}]
    measured(router, monkeypatch, counting=lambda *_: 756, contexts={MODEL: 820})

    result = router.generate(MODEL, messages, max_output_tokens=64)

    assert result.admission.input_count.tokens == 756
    assert result.admission.input_count.quality == "exact"
    assert result.admission.input_count.provenance == "synthetic chat template"
    assert result.admission.context_tokens == 820
    assert result.admission.output_tokens == 64
    assert calls[-1]["messages"] == messages
    assert calls[-1]["max_completion_tokens"] == 64
    with pytest.raises(gm.GenerationError, match="fit") as caught:
        router.generate(MODEL, messages, max_output_tokens=65)
    assert caught.value.code == "input_budget"
    assert len(calls) == 1


def test_only_complete_oldest_history_turn_is_removed(monkeypatch):
    router, calls = client([{"id": MODEL, "context_window": 201}])
    messages = [
        {"role": "system", "content": "keep system"},
        {"role": "user", "content": "old question " * 60},
        {"role": "assistant", "content": "old answer " * 60},
        {"role": "user", "content": "keep newer question"},
        {"role": "assistant", "content": "keep newer answer"},
        {"role": "user", "content": "keep the entire current query and evidence"},
    ]
    measured(router, monkeypatch, counting=lambda _, m: 300 if len(m) > 4 else 137,
             contexts={MODEL: 201})
    result = router.generate(MODEL, messages, max_output_tokens=64)
    assert calls[0]["messages"] == [messages[0], *messages[3:]]
    assert result.admission.history_turns_removed == 1
    assert len(messages) == 6


def test_large_request_does_not_hit_streamlit_fixed_ceiling(load_streamlit_app, monkeypatch):
    app, _ = load_streamlit_app
    source = sources()[0]
    source["citation_id"] = 1
    source["document"] = SOURCE + " extra fixture text " * 8000
    monkeypatch.setattr(app, "process_chat_mode", lambda *a, **k: ("", "", [source]))
    router, calls = client([{"id": MODEL, "context_window": 131072}],
                           lambda request: iter([delta(packet(1, SOURCE)), delta(finish="stop")]))
    measured(router, monkeypatch, counting=lambda *_: 30000, contexts={MODEL: 131072})
    prepared = app._prepare_generation(
        QUESTION, "Use the fixture.", 0.1, "Standard", 5, provider="groq",
        preferred=MODEL, router=router, strict=True, retrieve=None, conversation_history=[],
    )
    assert prepared.error is None
    from rag_prompting import generate_structural_answer
    result = generate_structural_answer(router, prepared.answer, provider="groq", **prepared.request)
    assert result.status == "structurally_validated"
    assert source["document"] in calls[0]["messages"][-1]["content"]


@pytest.mark.parametrize(("terminal", "status"), [
    ({"truncated": True}, "incomplete"),
    ({"x_groq": {"error": "synthetic upstream early stop"}}, "incomplete"),
    ({"x_groq": {"error": "prompt is longer than the context length"}}, "input_budget"),
    ({"x_groq": {"usage": {"prompt_tokens": 10, "completion_tokens": 15}}}, "incomplete"),
    ({"usage": {"prompt_tokens": 10, "completion_tokens": 15}}, "incomplete"),
])
def test_explicit_or_comparable_truncation_blocks_structural_publication(monkeypatch, terminal, status):
    from rag_prompting import build_numbered_sources, generate_structural_answer, prepare_structural_answer

    prepared = prepare_structural_answer("System", QUESTION, build_numbered_sources(sources()))
    end_packet = delta(finish="stop")
    end_packet.update(terminal)
    router, calls = client([{"id": MODEL, "context_window": 4096}],
                           lambda request: iter([delta(packet(1, SOURCE)), end_packet]))
    measured(router, monkeypatch, counting=lambda *_: 350, contexts={MODEL: 4096})
    result = generate_structural_answer(router, prepared, provider="groq", preferred=MODEL)
    assert result.status == status
    assert result.text == ""
    assert result.evidence == prepared.evidence
    assert calls[0]["messages"] == prepared.messages()


@pytest.mark.parametrize("fallback_limit", [763, 764])
def test_fallback_admits_identical_evidence_with_its_own_count(monkeypatch, fallback_limit):
    import groq
    import httpx
    from rag_prompting import build_numbered_sources, generate_structural_answer, prepare_structural_answer

    def send(request):
        if request["model"] == MODEL:
            response = httpx.Response(404, request=httpx.Request("POST", "https://example.invalid"))
            raise groq.NotFoundError("fixture", response=response, body={"error": {"code": "model_not_found"}})
        replacement_delta = delta(packet(1, SOURCE))
        replacement_stop = delta(finish="stop")
        for event in (replacement_delta, replacement_stop):
            event["model"] = FALLBACK
        return iter([replacement_delta, replacement_stop])

    router, calls = client([
        {"id": MODEL, "context_window": 8192},
        {"id": FALLBACK, "context_window": fallback_limit},
    ], send)
    evaluations = []
    measured(router, monkeypatch, counting=lambda model, _: 550 if model == MODEL else 700,
             contexts={MODEL: 8192, FALLBACK: fallback_limit}, seen=evaluations)
    prepared = prepare_structural_answer("System", QUESTION, build_numbered_sources(sources()))
    result = generate_structural_answer(router, prepared, provider="groq", preferred=MODEL,
                                        max_output_tokens=64)

    assert [call["model"] for call in calls] == ([MODEL, FALLBACK] if fallback_limit == 764 else [MODEL])
    assert all(call["messages"] == prepared.messages() for call in calls)
    assert all(call["max_completion_tokens"] == 64 for call in calls)
    assert (FALLBACK, prepared.messages()) in evaluations
    assert result.evidence.context_sha256 == prepared.evidence.context_sha256
    assert result.evidence.package_sha256 == prepared.evidence.package_sha256
    assert [(row["collection_id"], row["chunk_id"]) for row in result.evidence.sources] == [
        ("synthetic", "chunk-1"), ("synthetic", "chunk-2"),
    ]
    assert result.status == ("structurally_validated" if fallback_limit == 764 else "input_budget")
    assert bool(result.text) == (fallback_limit == 764)


@pytest.fixture
def tokenizer_assets(tmp_path):
    from hashlib import sha256

    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {symbol: i for i, symbol in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    # The request counter must explicitly undo any truncation serialized with the asset.
    tokenizer.enable_truncation(12)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    template_path = tmp_path / "template.jinja"
    template_path.write_text(
        '{% for m in messages %}<{{m.role}}>{{m.content}}</{{m.role}}>{% endfor %}'
        '{% if add_generation_prompt %}<assistant>{% endif %}', encoding="utf-8",
    )
    return {
        "provider": "groq", "model": MODEL, "revision": "synthetic-v1",
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path.read_bytes()).hexdigest(),
        "template_path": str(template_path),
        "template_sha256": sha256(template_path.read_bytes()).hexdigest(),
        "add_special_tokens": False, "max_added_tokens": 0,
    }


def render_fixture(messages):
    return "".join(
        f"<{message['role']}>{message['content']}</{message['role']}>"
        for message in messages
    ) + "<assistant>"


@pytest.mark.parametrize("extra, quality", [(0, "exact"), (17, "conservative")])
def test_reports_count_from_complete_verified_assets(tokenizer_assets, extra, quality):
    messages = [
        {"role": "system", "content": "Use only the serialized evidence."},
        {"role": "user", "content": "呼吸器内科を調べる"},
        {"role": "assistant", "content": "e\u0301 Å 🫁"},
        {"role": "user", "content": (
            "Question\nEvidence: N-acetylneuraminic acid and phosphatidylinositol-3-kinase\n"
            "def f(xs):\n    return [x ** 2 for x in xs]\n"
        )},
    ]
    rendered_count = len(render_fixture(messages).encode("utf-8"))
    total = rendered_count + extra + 64
    router, calls = client([{"id": MODEL, "revision": "synthetic-v1", "context_length": total}])
    spec = {**tokenizer_assets, "max_added_tokens": extra}
    router._profile_rows[MODEL]["admission"] = spec

    result = router.generate(MODEL, messages, max_output_tokens=64)

    assert result.admission.input_count.tokens == rendered_count + extra
    assert result.admission.input_count.quality == quality
    assert result.admission.context_tokens == total
    provenance = result.admission.input_count.provenance
    for identity in (MODEL, spec["revision"], spec["tokenizer_sha256"], spec["template_sha256"]):
        assert identity in provenance
    assert calls[0]["messages"] == messages


@pytest.mark.parametrize("case", [
    "wrong_model", "wrong_revision", "wrong_deployment_tokenizer", "wrong_tokenizer_digest",
    "missing_tokenizer", "missing_template", "wrong_template_digest", "unsafe_template",
])
def test_unverified_deployment_assets_never_dispatch(tokenizer_assets, tmp_path, case):
    from hashlib import sha256
    from pathlib import Path

    spec = tokenizer_assets
    row = {"id": MODEL, "context_window": 131072, "revision": "synthetic-v1"}
    if case == "wrong_model":
        spec["model"] = FALLBACK
    elif case == "wrong_revision":
        row["revision"] = "synthetic-v2"
    elif case == "wrong_deployment_tokenizer":
        row["tokenizer_sha256"] = "1" * 64
    elif case == "wrong_tokenizer_digest":
        spec["tokenizer_sha256"] = "2" * 64
    elif case == "missing_tokenizer":
        spec["tokenizer_path"] = str(tmp_path / "absent-tokenizer.json")
    elif case == "missing_template":
        spec["template_path"] = str(tmp_path / "absent-template.jinja")
    elif case == "wrong_template_digest":
        spec["template_sha256"] = "3" * 64
    else:
        changed_template = Path(spec["template_path"])
        changed_template.write_text("omitted system, question and evidence", encoding="utf-8")
        spec["template_sha256"] = sha256(changed_template.read_bytes()).hexdigest()
    router, calls = client([row])
    router._profile_rows[MODEL]["admission"] = spec

    with pytest.raises(gm.GenerationError) as caught:
        router.generate(MODEL, [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "Question with serialized evidence"},
        ], max_output_tokens=64)
    assert caught.value.code == "input_budget"
    assert caught.value.admission.input_count.quality == "unknown"
    assert calls == []


def test_groq_template_sees_effective_reasoning_options(tokenizer_assets):
    from hashlib import sha256
    from pathlib import Path

    template = Path(tokenizer_assets["template_path"])
    template.write_text(
        "{% if reasoning_effort == 'low' and not include_reasoning %}{{messages[0].content}}"
        "{% endif %}", encoding="utf-8",
    )
    spec = {**tokenizer_assets, "model": FALLBACK, "template_sha256": sha256(template.read_bytes()).hexdigest()}
    router, calls = client([{"id": FALLBACK, "context_window": 4096}])
    router._profile_rows[FALLBACK]["admission"] = spec
    router._client.chat.completions.create = lambda **kwargs: (
        calls.append(kwargs) or iter([{"model": FALLBACK, "choices": [{
            "delta": {"content": "ok"}, "finish_reason": "stop",
        }]}])
    )
    result = router.generate(FALLBACK, [{"role": "user", "content": "糖"}], max_output_tokens=64)
    assert result.admission.input_count.tokens == 3
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["include_reasoning"] is False


def local_router(assets, responder, *, contexts=(8192, 8192, 8192), model_limit=32768,
                 version="0.34.0", digest="d" * 64):
    from hashlib import sha256
    import json

    import httpx
    import ollama

    model = "phi-3"
    template = "synthetic server template"
    spec = {
        **assets, "provider": "ollama", "model": model, "server_version": "0.34.0",
        "model_digest": "d" * 64, "template_sha256": sha256(template.encode()).hexdigest(),
    }
    runner_contexts = iter(contexts)
    generation_calls = []
    render_calls = []

    def handle(request):
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": version})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"model": model, "digest": digest}]})
        if path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"], "template": template,
                                            "model_info": {"synthetic.context_length": model_limit}})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{
                "model": f"{model}:latest", "digest": digest, "context_length": next(runner_contexts),
            }]})
        assert path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["truncate"] is payload["shift"] is False
        rendered_prompt = render_fixture(payload["messages"])
        if payload.get("_debug_render_only"):
            render_calls.append(payload)
            return httpx.Response(200, json={
                "model": model, "debug_info": {"rendered_template": rendered_prompt, "image_count": 0},
            })
        generation_calls.append(payload)
        return responder(len(rendered_prompt.encode("utf-8")))

    router = gm.OllamaRouter(client=ollama.Client(
        host="http://localhost:11434", transport=httpx.MockTransport(handle),
        follow_redirects=False, trust_env=False,
    ))
    router._profile_rows[model]["admission"] = spec
    return router, render_calls, generation_calls


def test_ollama_uses_effective_allocation_without_lowering_runner(tokenizer_assets):
    import httpx

    def respond(count):
        return httpx.Response(200, json={
            "model": "phi-3:latest", "done": True, "done_reason": "stop",
            "message": {"content": "accepted"}, "prompt_eval_count": count,
            "prompt_eval_cached_count": count - 1, "eval_count": 1,
        })

    messages = [{"role": "user", "content": "糖" * 2000}]
    router, renders, calls = local_router(tokenizer_assets, respond)
    result = router.generate("phi-3", messages, stream=False, max_output_tokens=64,
                             context_tokens=7200, options={"num_ctx": 7500})

    assert result.admission.context_tokens == 7200
    assert result.admission.input_count.tokens == len(render_fixture(messages).encode("utf-8"))
    assert result.usage["cached_prompt_tokens"] + 1 == result.usage["prompt_tokens"]
    assert renders[0]["options"]["num_ctx"] == calls[0]["options"]["num_ctx"] == 8192
    assert calls[0]["options"]["num_predict"] == 64
    assert renders[0]["messages"] == calls[0]["messages"] == messages


@pytest.mark.parametrize("options, expected_renders", [
    ({"contexts": (None,)}, 0),
    ({"contexts": (4096, 4096)}, 1),
    ({"contexts": (8192, 4096)}, 1),
    ({"contexts": (8192, 8192, 4096)}, 1),
    ({"model_limit": None}, 0),
    ({"version": "0.33.0"}, 0),
    ({"digest": "0" * 64}, 0),
])
def test_ollama_refuses_unknown_smaller_or_changed_effective_capacity(
    tokenizer_assets, options, expected_renders,
):
    router, renders, calls = local_router(tokenizer_assets, lambda _: pytest.fail("dispatched"), **options)
    with pytest.raises(gm.GenerationError) as caught:
        router.generate("phi-3", [{"role": "user", "content": "星" * 2000}], max_output_tokens=64)
    assert caught.value.code == "input_budget"
    assert len(renders) == expected_renders
    assert calls == []


@pytest.mark.parametrize(("mode", "status"), [
    ("truncation", "incomplete"), ("length", "incomplete"), ("reset", "provider_failed"),
    ("overflow", "input_budget"), ("usage_loss", "incomplete"), ("normal", "structurally_validated"),
])
def test_ollama_transport_termination_gates_structural_answer(tokenizer_assets, mode, status):
    import json
    import httpx
    from rag_prompting import build_numbered_sources, generate_structural_answer, prepare_structural_answer

    prepared = prepare_structural_answer("System", QUESTION, build_numbered_sources(sources()))

    def encode(event):
        return (json.dumps({"model": "phi-3:latest", **event}) + "\n").encode()

    def respond(count):
        if mode == "overflow":
            return httpx.Response(400, json={"error": "the prompt is longer than the context length currently available"})
        first = encode({"done": False, "message": {"content": packet(1, SOURCE)}})

        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                yield first
                if mode == "reset":
                    raise httpx.ReadError("synthetic reset after answer bytes")
                yield encode({
                    "done": True, "done_reason": "length" if mode == "length" else "stop",
                    "prompt_truncated": mode == "truncation",
                    "prompt_eval_count": count - int(mode == "usage_loss"),
                    "prompt_eval_cached_count": count - 1, "eval_count": 10,
                })

        return httpx.Response(200, stream=Stream())

    router, _, calls = local_router(tokenizer_assets, respond)
    result = generate_structural_answer(router, prepared, provider="ollama", preferred="phi-3")
    assert result.status == status
    assert bool(result.text) is (mode == "normal")
    assert result.evidence == prepared.evidence
    assert len(calls) == 1
    assert calls[0]["messages"] == prepared.messages()
