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
