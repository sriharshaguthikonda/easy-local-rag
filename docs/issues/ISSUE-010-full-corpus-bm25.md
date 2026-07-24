# Issue #10: Search BM25 over the full corpus

[Roadmap ledger](README.md)

**Status:** OPEN — current GUI code appears to load a broader document set, but independent full-corpus behavior and cache invalidation need proof.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/10
**Labels / priority:** `priority:P1`, `type:perf`
**Dependencies:** Prefer `.memory` #5/#23 lexical replacement; create a dedicated legacy-cache packet only if legacy BM25 remains supported. #11 must reject model/collection mismatch before retrieval.

## Implementation slices

1. Read all collection documents and metadata, build a tokenized `BM25Okapi` index independently of vector-query results.
2. Cache by Chroma path and collection name plus a collection generation/fingerprint or explicit ingestion invalidation hook. Count may be diagnostic input but is not sufficient because content can change at the same count.
3. Query vector and BM25 independently, fuse with a deterministic method such as reciprocal-rank fusion, and preserve document/metadata for BM25-only hits.

## Affected interfaces, files, and artifacts

- `GUI_direct_search.py`, `GUI_context.py`, `GUI_chromadb.py`, optional `bm25_index.py`, tests.
- Retrieval contract: an exact-term hit absent from vector top-N may still appear in the fused result.

## Concrete actions

- Do not rebuild a BM25 corpus from the vector response.
- Use the existing `rank_bm25` dependency; avoid a new persistence layer until corpus rebuild cost proves it necessary.
- Test a fake collection in which vector returns A while BM25 recovers B.
- Replace one document while preserving collection count; assert the next query rebuilds or invalidates the cache and returns the updated term rather than stale content.

## Verification

```powershell
python -m pytest tests/test_gui_direct_search_bm25.py -q
python -m pytest tests -q
python -m py_compile GUI_direct_search.py
```

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** test evidence shows a BM25-only hit joins fused results, full-corpus loading is independent, and generation/fingerprint/hook invalidation passes a same-count content-update regression.
- **Retirement closure (mutually exclusive):** remove BM25 claims and controls from supported paths/docs, prove the stale cache path cannot run, and document the maintained retrieval replacement without describing vector-top-N reranking as full-corpus BM25.
- **Rollback constraint:** retain vector retrieval as a working fallback but do not label vector-top-N reranking as full-corpus BM25.
- **Commit:** `fix(#10): search BM25 over full corpus`.
