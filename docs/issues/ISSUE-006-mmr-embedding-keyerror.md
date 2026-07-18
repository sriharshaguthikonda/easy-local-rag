# Issue #6: Fix hybrid-retrieval MMR embedding access

**Status:** OPEN — partial code repair exists: query includes `embeddings` and MMR reads top-level embeddings. The approved functional fake-collection regression and document-array mapping are still required.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/6
**Labels / priority:** `priority:P0`, `type:bug`
**Dependencies:** Preserve the #11 embedding-model contract; use a fake collection rather than a live Chroma database in the regression test.

## Implementation slices

1. Build vector results by zipping `metadatas`, `distances`, `embeddings`, **and** `documents` from Chroma's parallel result arrays. Keep `embedding` and `document` top-level; metadata contains only metadata.
2. Give keyword-only hits `embedding=None`; MMR computes dot products only when both operands are present, while preserving the `(context, metadata)` return shape.
3. Replace the static source-string assertion with the approved functional fake-collection test: its metadata excludes embeddings, its documents array supplies chunk text, and `additional_unique_files > 0` exercises MMR.

## Affected interfaces, files, and artifacts

- `streamlit_groq_lama_chromadb_RAG_ETTS.py::get_relevant_context_hybrid`.
- `tests/test_streamlit_hybrid_embeddings.py` and its fake collection/query payload.
- Chroma query contract: `include=["documents", "metadatas", "distances", "embeddings"]` and parallel first-result arrays.

## Concrete actions

- Do not read `meta["embedding"]`; do not rely on `meta["text"]` when Chroma's `documents` array is the canonical chunk source.
- Stub rewrite/embedding dependencies in the test, then assert selected context includes a document-array value, no `KeyError` occurs, and the result is not the fallback string.
- Keep broad failure handling from hiding regressions: test the normal path and retain a stable empty/error return on expected failure.

## Verification

```powershell
python -m pytest tests/test_streamlit_hybrid_embeddings.py -q
python -m pytest tests -q
python -m py_compile streamlit_groq_lama_chromadb_RAG_ETTS.py
```

## Closure gate, rollback, and commit boundary

- **Close only when:** the fake collection proves metadata without embeddings and document-array content work with `additional_unique_files > 0`; manual Streamlit search shows sources without the swallowed fallback.
- **Rollback constraint:** retain the `embeddings` include and top-level result shape; never restore embedding-in-metadata access.
- **Commit:** `fix(#6): use Chroma embeddings in MMR`.
