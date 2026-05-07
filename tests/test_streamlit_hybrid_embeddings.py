from pathlib import Path


def test_streamlit_hybrid_queries_embeddings_array():
    source = Path("streamlit_groq_lama_chromadb_RAG_ETTS.py").read_text(
        encoding="utf-8"
    )
    assert 'include=["documents", "metadatas", "distances", "embeddings"]' in source
    assert '"embedding": emb' in source
    assert 'res["meta"]["embedding"]' not in source
