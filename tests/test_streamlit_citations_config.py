from pathlib import Path
from types import SimpleNamespace

import pytest

from test_generation_streamlit import load_streamlit_app


def _record_retrieval(result):
    calls = []

    def retrieve(prompt, **options):
        calls.append((prompt, options))
        if isinstance(result, Exception):
            raise result
        return result

    return retrieve, calls


class _DisplayContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _DisplayStreamlit:
    def __init__(self):
        self.markdowns = []
        self.plain_text = []
        self.buttons = []

    def expander(self, *_args, **_kwargs):
        return _DisplayContext()

    def markdown(self, text):
        self.markdowns.append(text)

    def text(self, text):
        self.plain_text.append(text)

    def columns(self, _widths):
        return _DisplayContext(), _DisplayContext()

    def button(self, label, key):
        self.buttons.append((label, key))
        return False


def test_source_preview_never_renders_evidence_as_markdown(load_streamlit_app):
    app, _display = load_streamlit_app
    display = _DisplayStreamlit()
    app.st = display
    filename = "synthetic[sample].txt"
    excerpt = "![synthetic image](https://example.invalid/evidence-marker)"

    app.render_source_documents(
        [{"citation_id": 1, "file_name": filename, "document": excerpt}], [1]
    )

    assert filename in "\n".join(display.plain_text)
    assert excerpt in display.plain_text
    assert excerpt not in "\n".join(display.markdowns)


@pytest.mark.parametrize("mode", ["Standard", "Focused Search", "Brain Dump", "Summary"])
def test_every_streamlit_mode_sends_retrieved_text_with_stable_citations(
    mode, load_streamlit_app
):
    sentinel = "SENTINEL evidence reaches the model prompt."
    retrieved = [
        {
            "file_name": "one.txt", "document": sentinel, "score": 0.9,
            "collection_id": "first", "chunk_id": "shared-chunk",
        },
        {
            "file_name": "two.txt", "document": "second source", "score": 0.8,
            "collection_id": "second", "chunk_id": "shared-chunk",
        },
    ]
    app, _display = load_streamlit_app
    retrieve, calls = _record_retrieval((f"{sentinel}\n\nsecond source", retrieved))

    prepared = app._prepare_generation(
        "question", "Trusted application policy.", 0.7, mode, 5,
        provider="ollama", preferred=None, router=None, strict=False,
        retrieve=retrieve, conversation_history=[],
    )
    model_prompt = prepared.request["messages"][-1]["content"]

    assert len(calls) == 1
    assert prepared.context == f"{sentinel}\n\nsecond source"
    assert prepared.error is None
    assert model_prompt.count(sentinel) == 1
    assert '<source id="1">' in model_prompt
    assert '<source id="2">' in model_prompt
    assert [
        (source["citation_id"], source["document"], source["collection_id"], source["chunk_id"])
        for source in prepared.metadata
    ] == [
        (1, sentinel, "first", "shared-chunk"),
        (2, "second source", "second", "shared-chunk"),
    ]


def test_joined_context_cannot_diverge_from_structured_sources(load_streamlit_app):
    app, _display = load_streamlit_app
    retrieve, _calls = _record_retrieval(
        ("wrong text", [{"file_name": "one.txt", "document": "actual text"}])
    )

    with pytest.raises(ValueError, match="does not match structured sources"):
        app.retrieve_numbered_sources("question", retrieve=retrieve)


def test_retrieved_sentinel_reaches_model_request_exactly_once(load_streamlit_app):
    app, _display = load_streamlit_app
    sentinel = "SENTINEL evidence reaches the model request."
    retrieve, _calls = _record_retrieval(
        (sentinel, [{"file_name": "one.txt", "document": sentinel}])
    )
    model_calls = []

    def generate(**request):
        model_calls.append(request)
        return SimpleNamespace(text="Answer [1].", actual_model="test-model", reason=None)

    _context, response, sources = app.chat_with_model(
        "question", "system", preferred="test-model",
        router=SimpleNamespace(generate=generate), retrieve=retrieve,
    )

    assert response == "Answer [1]."
    assert len(model_calls) == 1
    assert model_calls[0]["messages"][-1]["content"].count(sentinel) == 1
    assert [(source["citation_id"], source["document"]) for source in sources] == [
        (1, sentinel)
    ]


@pytest.mark.parametrize("retrieval_result", [RuntimeError("sensitive details"), ("", [])])
def test_missing_evidence_is_visible_and_skips_model_providers(
    load_streamlit_app, retrieval_result
):
    app, display = load_streamlit_app
    retrieve, _calls = _record_retrieval(retrieval_result)
    router = SimpleNamespace(generate=lambda **kwargs: pytest.fail("Unexpected model request"))

    result = app.chat_with_model("question", "system", retrieve=retrieve, router=router)

    assert result == (None, None, None)
    assert display.errors == ["No usable sources were retrieved for this request."]


def test_oversized_evidence_prompt_is_visible_and_skips_model_providers(load_streamlit_app):
    app, display = load_streamlit_app
    document = "evidence " * 6000
    retrieve, _calls = _record_retrieval(
        (document, [{"file_name": "oversized.txt", "document": document}])
    )
    router = SimpleNamespace(generate=lambda **kwargs: pytest.fail("Unexpected model request"))

    context, answer, sources = app.chat_with_model(
        "question", "system", retrieve=retrieve, router=router
    )

    assert (context, answer) == (document, None)
    assert sources[0]["document"] == document
    assert display.errors == ["The full evidence does not fit the input budget."]


def test_source_renderer_uses_citation_ids_and_document_excerpt(load_streamlit_app):
    app, _ = load_streamlit_app
    display = _DisplayStreamlit()
    app.st = display

    app.render_source_documents(
        [
            {
                "citation_id": 7,
                "file_name": "folder/example.txt",
                "document": "Evidence excerpt.",
            }
        ],
        [7],
    )

    assert display.plain_text == ["[7] example.txt", "Evidence excerpt."]
    assert display.markdowns == ["**Excerpt:**", "---"]
    assert [key for _label, key in display.buttons] == [
        f"open_7_{hash(str(Path('folder/example.txt')))}",
        f"copy_7_{hash(str(Path('folder/example.txt')))}",
    ]
