from pathlib import Path


def test_bm25_uses_full_collection_get():
    source = Path("GUI_direct_search.py").read_text(encoding="utf-8")
    assert 'full = col.get(include=["documents","metadatas"])' in source
    assert (
        'bm25_corpus = [doc or "" for doc in res.get("documents",[[]])[0]]'
        not in source
    )
