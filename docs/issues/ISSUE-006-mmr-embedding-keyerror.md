# Issue #6: Fix hybrid-retrieval MMR embedding access

**Status:** OPEN — implementation merged to the GUI lineage in [PR #29](https://github.com/sriharshaguthikonda/easy-local-rag/pull/29) at fix commit [`de74659`](https://github.com/sriharshaguthikonda/easy-local-rag/commit/de7465902e00e363993f9ea9eb2190ce940e1bed); close after this canonical plan lands on `main` and the issue receives reciprocal evidence links.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/6
**Labels / priority:** `priority:P0`, `type:bug`
**Dependencies:** Preserve the #11 embedding-model contract; use a fake collection rather than a live Chroma database in the regression test.

## Implementation slices

1. Build vector results by zipping `metadatas`, `distances`, `embeddings`, **and** `documents` from Chroma's parallel result arrays. Keep `embedding` and `document` top-level; metadata contains only metadata.
2. Seed MMR with the top-K results, rescore remaining candidates against every selected embedding, append one best candidate per iteration, and preserve the `(context, metadata)` return shape.
3. Replace the static source-string assertion with the approved functional fake-collection test: its metadata excludes text and embeddings, its documents array supplies chunk text, and `additional_unique_files > 0` distinguishes greedy incremental MMR from a one-shot relevance sort.

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
python -m pytest tests/test_streamlit_hybrid_embeddings.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider
python -m py_compile streamlit_groq_lama_chromadb_RAG_ETTS.py
```

## Closure gate, rollback, and commit boundary

- **Close only when:** the focused fake-collection regression passes with metadata containing only filename/time, document-array content, separate embeddings, `top_k > 0`, and `additional_unique_files > 0`; the backend compiles; PR #29 is merged; and the closing comment links this main-line plan, exact fix commit, and verification evidence.
- **Rollback constraint:** retain the `embeddings` include and top-level result shape; never restore embedding-in-metadata access.
- **Commit:** `fix(#6): use Chroma documents in hybrid retrieval`.
