from types import SimpleNamespace

import pytest

from test_generation_streamlit import load_streamlit_app


@pytest.mark.parametrize("provider, should_speak", [("groq", True), ("ollama", False)])
def test_streamlit_queues_completed_answer_only_for_cloud_speech(
    load_streamlit_app, monkeypatch, provider, should_speak
):
    app, display = load_streamlit_app
    display.session_state.allow_cloud_speech = True
    spoken = []
    monkeypatch.setattr(app, "ensure_tts_worker", lambda: {"queue": SimpleNamespace(put=spoken.append)})

    def generate(**request):
        request["on_chunk"]("Answer ")
        assert spoken == []
        request["on_chunk"]("[1].")
        assert spoken == []
        return SimpleNamespace(text="Answer [1].", actual_model="selected", reason=None)

    _context, answer, _sources = app.chat_with_model(
        "question", "system", provider=provider, router=SimpleNamespace(generate=generate),
        retrieve=lambda *args, **options: ("evidence", [{"document": "evidence"}]),
    )

    assert answer == "Answer [1]."
    assert spoken == (["Answer [1]."] if should_speak else [])
