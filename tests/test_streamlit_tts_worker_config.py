from pathlib import Path


def test_streamlit_uses_cached_tts_worker_resource():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "@st.cache_resource" in source
    assert "def get_tts_worker_resource()" in source
    assert "st.session_state.tts_queue.put(current_sentence.strip())" in source
