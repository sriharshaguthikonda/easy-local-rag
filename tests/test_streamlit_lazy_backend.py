from pathlib import Path


def test_streamlit_uses_lazy_backend_import():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "RAG_BACKEND_IMPORT_ERROR = None" in source
    assert "import streamlit_groq_lama_chromadb_RAG_ETTS as rag_backend" in source
    assert "def _require_rag_backend()" in source


def test_streamlit_handles_collection_init_failure():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'st.session_state.collection = None' in source
    assert "collection_init_error" in source
