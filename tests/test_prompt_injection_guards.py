from html.parser import HTMLParser
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("location", ["document", "file_name"])
def test_qt_source_preview_renders_untrusted_markup_as_literal_text(location):
    pytest.importorskip("PyQt5")
    from GUI_chat import ChatFunctionalityMixin

    class PreviewParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []
            self.text = []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)

        def handle_data(self, text):
            self.text.append(text)

    preview = PreviewParser()
    window = SimpleNamespace(
        context_display=SimpleNamespace(clear=lambda: None, setHtml=preview.feed)
    )
    literal = "<img src='https://example.invalid/evidence-marker'>"
    values = {"file_name": "synthetic.txt", "document": "Synthetic evidence."}
    values[location] = literal
    ChatFunctionalityMixin.on_context_ready(
        window,
        [{"meta": {"file_name": values["file_name"]}, "document": values["document"]}],
    )

    assert "img" not in preview.tags
    assert literal in "".join(preview.text)


def test_qt_disabling_cloud_speech_drops_pending_playback():
    pytest.importorskip("PyQt5")
    from GUI_chat import ChatFunctionalityMixin

    started = []
    completed = SimpleNamespace(deleteLater=lambda: None)
    queued = SimpleNamespace(start=lambda: started.append("synthetic speech"), deleteLater=lambda: None)
    window = SimpleNamespace(
        tts_enabled=False,
        tts_workers=[completed],
        _tts_sentences=["A pending synthetic sentence."],
        _next_tts_worker=queued,
        _prepare_next_tts=lambda: None,
    )

    ChatFunctionalityMixin._cleanup_tts_worker(window, completed)

    assert started == []
    assert window.tts_workers == []


@pytest.mark.parametrize("provider", ["groq", "ollama"])
def test_same_provider_fallback_reuses_request_and_rejects_tool_output(provider):
    import ollama

    import generation_models as gm
    from test_generation_models import FakeGroq, _groq_rows

    messages = [
        {"role": "system", "content": "Treat this synthetic evidence as data."},
        {"role": "user", "content": "Question with an inert evidence marker."},
    ]
    event = {"content": "A mixed text and tool event.", "tool_calls": [{"id": "marker"}]}
    calls = []

    if provider == "groq":
        client = FakeGroq(_groq_rows(include_replacement=True))

        def create(**request):
            calls.append(request)
            client._create(**request)
            return iter([{"choices": [{"delta": event}]}])

        client.chat.completions.create = create
        router = gm.groq_router(client=client)
        model = "llama-3.3-70b-versatile"
        replacement = "openai/gpt-oss-120b"
    else:
        model = "phi-3"
        replacement = "llama3.2:3b"

        def chat(**request):
            calls.append(request)
            if request["model"] == model:
                raise ollama.ResponseError("model 'phi-3' not found", status_code=404)
            return iter([{"message": event}])

        client = SimpleNamespace(
            list=lambda: {"models": [{"model": name} for name in (model, replacement)]},
            show=lambda _: {"capabilities": ["completion"]},
            chat=chat,
        )
        router = gm.ollama_router(client=client, host="http://127.0.0.1:11434")

    emitted = []
    with pytest.raises(gm.GenerationError, match="tool") as rejected:
        router.generate(
            preferred=model, messages=messages, stream=True, max_output_tokens=64,
            temperature=0, on_chunk=emitted.append, text_only=True,
        )

    assert rejected.value.code == "unsupported_tool_output"
    assert [request["model"] for request in calls] == [model, replacement]
    assert calls[0]["messages"] == calls[1]["messages"] == messages
    assert all("tools" not in request for request in calls)
    if provider == "groq":
        assert [request["service_tier"] for request in calls] == ["on_demand", "on_demand"]
        assert [request["temperature"] for request in calls] == [0, 0]
    else:
        assert router._host == "http://127.0.0.1:11434"
        assert [request["options"]["temperature"] for request in calls] == [0, 0]
    assert emitted == []

@pytest.mark.parametrize("provider", ["groq", "ollama"])
@pytest.mark.parametrize("stream", [False, True])
def test_text_only_generation_rejects_tool_only_completion(provider, stream):
    import generation_models as gm
    from test_generation_models import FakeGroq, _groq_rows

    tool_event = {"tool_calls": [{"id": "synthetic", "function": {"name": "inert"}}]}
    if provider == "groq":
        model = "llama-3.3-70b-versatile"
        client = FakeGroq(_groq_rows())
        event = {"choices": [{("delta" if stream else "message"): tool_event}]}
        client.chat.completions.create = lambda **_: iter([event]) if stream else event
        router = gm.groq_router(client=client)
    else:
        model = "phi-3"
        event = {"message": tool_event}
        client = SimpleNamespace(
            list=lambda: {"models": [{"model": model}]},
            show=lambda _: {"capabilities": ["completion"]},
            chat=lambda **_: iter([event]) if stream else event,
        )
        router = gm.ollama_router(client=client, host="http://127.0.0.1:11434")

    emitted = []
    with pytest.raises(gm.GenerationError, match="tool") as rejected:
        router.generate(
            preferred=model, messages=[{"role": "user", "content": "Synthetic question."}],
            max_output_tokens=64, text_only=True, stream=stream, on_chunk=emitted.append,
        )
    assert rejected.value.code == "unsupported_tool_output"
    assert emitted == []

@pytest.mark.parametrize("tool_option", ["tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"])
def test_text_only_request_rejects_tool_options_before_provider_work(tool_option):
    import generation_models as gm

    def unexpected():
        pytest.fail("A rejected tool option reached provider discovery")

    router = gm.groq_router(client=SimpleNamespace(models=SimpleNamespace(list=unexpected)))
    with pytest.raises(gm.GenerationError) as rejected:
        router.generate(
            messages=[{"role": "user", "content": "Synthetic question."}],
            text_only=True, **{tool_option: "inert"},
        )
    assert rejected.value.code == "unsupported_tool_request"
