from types import SimpleNamespace

import pytest

from structural_answer_fixtures import completed_excerpt, displayed_excerpt
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
        assert request.get("on_chunk") is None
        assert spoken == [] and display.plain_text == []
        return completed_excerpt("evidence")

    _context, answer, _sources = app.chat_with_model(
        "question", "system", provider=provider, router=SimpleNamespace(generate=generate),
        retrieve=lambda *args, **options: ("evidence", [{"document": "evidence"}]),
    )

    assert answer == displayed_excerpt("evidence")
    assert spoken == ([answer] if should_speak else [])
