import ast
from pathlib import Path

import pytest

from rag_prompting import (
    CONTEXT_GUARD,
    build_guarded_context_block,
    build_numbered_sources,
    context_from_sources,
)


def _load_prompt_functions(retrieval_result):
    tree = ast.parse(Path("streamlit_app.py").read_text(encoding="utf-8"))
    wanted = {"retrieve_numbered_sources", "build_citation_prompt", "process_chat_mode"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {function.name for function in functions} == wanted

    calls = []

    def retrieve(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return retrieval_result

    namespace = {
        "CONTEXT_GUARD": CONTEXT_GUARD,
        "build_guarded_context_block": build_guarded_context_block,
        "build_numbered_sources": build_numbered_sources,
        "context_from_sources": context_from_sources,
        "get_relevant_context_hybrid": retrieve,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "streamlit_app.py", "exec"), namespace)
    return namespace, calls


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
