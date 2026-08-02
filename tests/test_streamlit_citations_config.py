import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_prompting import (
    CONTEXT_GUARD,
    build_guarded_context_block,
    build_numbered_sources,
    context_from_sources,
    validate_response_citations,
)
from token_budget import trim_messages_to_budget


def _load_prompt_functions(retrieval_result):
    tree = ast.parse(Path("streamlit_app.py").read_text(encoding="utf-8"))
    wanted = {
        "retrieve_numbered_sources",
        "build_citation_prompt",
        "process_chat_mode",
        "chat_with_model",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {function.name for function in functions} == wanted

    calls = []

    def retrieve(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if isinstance(retrieval_result, BaseException):
            raise retrieval_result
        return retrieval_result

    namespace = {
        "CONTEXT_GUARD": CONTEXT_GUARD,
        "build_guarded_context_block": build_guarded_context_block,
        "build_numbered_sources": build_numbered_sources,
        "context_from_sources": context_from_sources,
        "get_relevant_context_hybrid": retrieve,
        "validate_response_citations": validate_response_citations,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "streamlit_app.py", "exec"), namespace)
    return namespace, calls


class _SessionState(dict):
    __getattr__ = dict.__getitem__


class _Placeholder:
    def markdown(self, _text):
        pass


@pytest.mark.parametrize("mode", ["Standard", "Focused Search", "Brain Dump", "Summary"])
def test_every_streamlit_mode_sends_retrieved_text_with_stable_citations(mode):
    sentinel = "SENTINEL evidence reaches the model prompt."
    retrieved = [
        {"file_name": "one.txt", "document": sentinel, "score": 0.9},
        {"file_name": "two.txt", "document": "second source", "score": 0.8},
    ]
    namespace, calls = _load_prompt_functions((f"{sentinel}\n\nsecond source", retrieved))

    context, model_prompt, sources = namespace["process_chat_mode"](
        "question", mode, context_window=5
    )

    assert len(calls) == 1
    assert context == f"{sentinel}\n\nsecond source"
    assert model_prompt.count(sentinel) == 1
    assert '<source id="1" file="one.txt">' in model_prompt
    assert '<source id="2" file="two.txt">' in model_prompt
    assert [(source["citation_id"], source["document"]) for source in sources] == [
        (1, sentinel),
        (2, "second source"),
    ]


def test_joined_context_cannot_diverge_from_structured_sources():
    namespace, _calls = _load_prompt_functions(
        ("wrong text", [{"file_name": "one.txt", "document": "actual text"}])
    )

    with pytest.raises(ValueError, match="does not match structured sources"):
        namespace["retrieve_numbered_sources"]("question")


def test_retrieved_sentinel_reaches_model_request_exactly_once():
    sentinel = "SENTINEL evidence reaches the model request."
    namespace, _calls = _load_prompt_functions(
        (sentinel, [{"file_name": "one.txt", "document": sentinel}])
    )
    model_calls = []

    def create_completion(**kwargs):
        model_calls.append(kwargs)
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Answer [1]."))]
            )
        ]

    namespace.update(
        st=SimpleNamespace(
            session_state=_SessionState(
                conversation_history=[],
                tts_enabled=False,
                tts_queue=SimpleNamespace(put=lambda _value: None),
            ),
            empty=lambda: _Placeholder(),
            info=lambda _message: None,
            error=lambda error: pytest.fail(str(error)),
        ),
        groq_client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_completion)
            )
        ),
        groq_model="test-model",
        ollama_model="fallback-model",
        trim_messages_to_budget=lambda messages, **_kwargs: (messages, False),
    )

    _context, response, sources = namespace["chat_with_model"](
        "question", "system"
    )

    assert response == "Answer [1]."
    assert len(model_calls) == 1
    assert model_calls[0]["messages"][-1]["content"].count(sentinel) == 1
    assert [(source["citation_id"], source["document"]) for source in sources] == [
        (1, sentinel)
    ]


def test_retrieval_failure_is_visible_and_skips_model_providers():
    namespace, _calls = _load_prompt_functions(RuntimeError("retrieval unavailable"))
    errors = []
    provider_calls = []
    namespace.update(
        st=SimpleNamespace(error=errors.append),
        groq_client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: provider_calls.append("groq")
                )
            )
        ),
        ollama=SimpleNamespace(
            chat=lambda **_kwargs: provider_calls.append("ollama")
        ),
    )

    result = namespace["chat_with_model"]("question", "system")

    assert result == (None, None, None)
    assert errors == ["An error occurred: retrieval unavailable"]
    assert provider_calls == []


def test_empty_retrieval_is_visible_and_skips_model_providers():
    namespace, _calls = _load_prompt_functions(("", []))
    errors = []
    provider_calls = []
    namespace.update(
        st=SimpleNamespace(
            session_state=_SessionState(
                conversation_history=[],
                tts_enabled=False,
                tts_queue=SimpleNamespace(put=lambda _value: None),
            ),
            error=errors.append,
            empty=lambda: _Placeholder(),
            info=lambda _message: None,
        ),
        groq_client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: provider_calls.append("groq")
                )
            )
        ),
        ollama=SimpleNamespace(
            chat=lambda **_kwargs: provider_calls.append("ollama")
        ),
        groq_model="test-model",
        ollama_model="fallback-model",
        trim_messages_to_budget=lambda messages, **_kwargs: (messages, False),
    )

    result = namespace["chat_with_model"]("question", "system")

    assert result == (None, None, None)
    assert errors == ["An error occurred: Retrieved sources are empty"]
    assert provider_calls == []


def test_oversized_evidence_prompt_is_visible_and_skips_model_providers():
    document = "evidence " * 6000
    namespace, _calls = _load_prompt_functions(
        (document, [{"file_name": "oversized.txt", "document": document}])
    )
    errors = []
    provider_calls = []
    namespace.update(
        st=SimpleNamespace(
            session_state=_SessionState(
                conversation_history=[],
                tts_enabled=False,
                tts_queue=SimpleNamespace(put=lambda _value: None),
            ),
            error=errors.append,
            empty=lambda: _Placeholder(),
            info=lambda _message: None,
        ),
        groq_client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: provider_calls.append("groq")
                )
            )
        ),
        ollama=SimpleNamespace(
            chat=lambda **_kwargs: provider_calls.append("ollama")
        ),
        groq_model="test-model",
        ollama_model="fallback-model",
        trim_messages_to_budget=trim_messages_to_budget,
    )

    result = namespace["chat_with_model"]("question", "system")

    assert result == (None, None, None)
    assert errors == ["An error occurred: Evidence prompt was truncated to fit token budget"]
    assert provider_calls == []
