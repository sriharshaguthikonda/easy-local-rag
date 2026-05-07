from pathlib import Path


def test_streamlit_prompt_includes_numbered_citation_blocks():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "def build_numbered_sources" in source
    assert "Use [N] citations for every factual claim" in source
    assert 'f"[{source[\'citation_id\']}] {source[\'source_name\']}\\n{source[\'text\']}"' in source


def test_streamlit_warns_when_citations_missing():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "No citations returned in response." in source
