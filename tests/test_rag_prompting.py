import pytest

from rag_prompting import (
    build_guarded_context_block,
    build_numbered_sources,
    context_from_sources,
    sanitize_context_text,
    validate_response_citations,
)


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


def test_numbered_sources_keep_document_mapping_and_order():
    sources = build_numbered_sources(
        [
            {"file_name": "folder/doc1.txt", "document": "alpha", "score": 0.9},
            {"file_name": "doc2.txt", "document": "beta", "score": 0.8},
        ]
    )

    assert [(source["citation_id"], source["source_name"], source["document"]) for source in sources] == [
        (1, "doc1.txt", "alpha"),
        (2, "doc2.txt", "beta"),
    ]
    assert context_from_sources(sources) == "alpha\n\nbeta"


@pytest.mark.parametrize("document", [None, "", "   "])
def test_numbered_sources_reject_missing_document_text(document):
    with pytest.raises(ValueError, match="missing document text"):
        build_numbered_sources([{"file_name": "doc.txt", "document": document}])


def test_citation_validation_rejects_ids_not_in_sent_evidence():
    sources = [{"citation_id": 1}, {"citation_id": 2}]

    assert validate_response_citations("Supported [2] and [1].", sources) == [1, 2]
    with pytest.raises(ValueError, match=r"unknown source IDs: 3"):
        validate_response_citations("Unsupported [3].", sources)
