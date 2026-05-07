from pathlib import Path


def test_streamlit_uses_context_guard():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "CONTEXT_GUARD" in source
    assert "build_guarded_context_block" in source


def test_gui_worker_uses_context_guard():
    source = Path("GUI_workers.py").read_text(encoding="utf-8")
    assert "CONTEXT_GUARD" in source
    assert "build_guarded_context_block" in source
