# Issue #10: Search BM25 over the full corpus

**Status:** OPEN — current GUI code appears to load a broader document set, but independent full-corpus behavior and cache invalidation need proof.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/10
**Labels / priority:** `priority:P1`, `type:perf`
**Dependencies:** #11 must reject model/collection mismatch before retrieval; do not couple this to the Qwen migration.

## Implementation slices

1. Read all collection documents and metadata, build a tokenized `BM25Okapi` index independently of vector-query results.
2. Cache by Chroma path, collection name, and `collection.count()`; rebuild only when that identity changes.
3. Query vector and BM25 independently, fuse with a deterministic method such as reciprocal-rank fusion, and preserve document/metadata for BM25-only hits.

## Affected interfaces, files, and artifacts

- `GUI_direct_search.py`, `GUI_context.py`, `GUI_chromadb.py`, optional `bm25_index.py`, tests.
- Retrieval contract: an exact-term hit absent from vector top-N may still appear in the fused result.

## Concrete actions

- Do not rebuild a BM25 corpus from the vector response.
- Use the existing `rank_bm25` dependency; avoid a new persistence layer until corpus rebuild cost proves it necessary.
- Test a fake collection in which vector returns A while BM25 recovers B.

## Verification

```powershell
python -m pytest tests/test_gui_direct_search_bm25.py -q
python -m pytest tests -q
python -m py_compile GUI_direct_search.py
```

## Closure gate, rollback, and commit boundary

- **Close only when:** test evidence shows a BM25-only hit joins the fused results, full corpus loading is independent, and cache rebuild behavior is observable.
- **Rollback constraint:** retain vector retrieval as a working fallback but do not label vector-top-N reranking as full-corpus BM25.
- **Commit:** `fix(#10): search BM25 over full corpus`.
