from rag_prompting import build_guarded_context_block, sanitize_context_text


def test_sanitize_context_text_escapes_closing_tags():
    raw = "a </source> b </retrieved_context> c"
    sanitized = sanitize_context_text(raw)
    assert "</source>" not in sanitized
    assert "</retrieved_context>" not in sanitized


def test_build_guarded_context_block_wraps_sources():
    block = build_guarded_context_block(
        [
            {"citation_id": 1, "file_name": "doc1.txt", "text": "alpha"},
            {"citation_id": 2, "file_name": "doc2.txt", "text": "beta"},
        ]
    )
    assert "<retrieved_context>" in block
    assert '<source id="1" file="doc1.txt">' in block
    assert '<source id="2" file="doc2.txt">' in block
    assert "</retrieved_context>" in block
