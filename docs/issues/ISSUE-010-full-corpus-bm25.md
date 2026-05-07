# Issue #10 Plan: Build BM25 over full corpus

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/10

Priority: P1 performance/search quality

## Goal

Make hybrid search actually hybrid. BM25 must search the full corpus
independently, not only rerank the top 50 vector hits.

## Files to inspect

- `GUI_direct_search.py`
- `GUI_context.py`
- `GUI_chromadb.py`
- optional new file: `bm25_index.py`
- tests under `tests/`

## Current bug shape

`GUI_direct_search.py` builds:

```python
bm25_corpus = [doc or "" for doc in res.get("documents", [[]])[0]]
```

But `res` already came from vector search. This means BM25 can never recover
documents that vector search missed.

## Implementation steps

1. Add a `BM25CorpusIndex` helper, preferably in `bm25_index.py`.
2. The helper should load all documents from Chroma:

   ```python
   collection.get(include=["documents", "metadatas"])
   ```

3. Build tokenized corpus from every document, not from vector results.
4. Cache the BM25 index by collection identity and document count. Minimum cache
   key:

   - Chroma path
   - collection name
   - result of `collection.count()`

5. Query vector search and BM25 independently:

   - vector: existing `collection.query(...)`
   - BM25: full corpus index top N

6. Fuse results. Preferred simple method: Reciprocal Rank Fusion:

   ```python
   score += 1 / (60 + rank)
   ```

   Keep existing weighted scoring only if tests prove it is stable.

7. Preserve metadata and document text in final results.
8. Avoid loading giant corpora repeatedly. Build once per worker process or cache
   inside the subprocess script.
9. If full corpus load is too large, add a documented cap and warning, but do
   not silently fall back to vector-only BM25.

## Tests and verification

Add a fake collection test:

- corpus has three docs
- vector query returns only doc A
- BM25 query for an exact term should return doc B
- final fused results include doc B

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile bm25_index.py GUI_direct_search.py
```

Manual smoke:

1. Add a document with a rare exact keyword.
2. Ask a query containing that keyword.
3. Confirm the document appears even if vector search rank alone would miss it.

## Acceptance checklist

- [ ] BM25 corpus comes from full Chroma collection.
- [ ] Vector and BM25 searches run independently.
- [ ] Fusion can include BM25-only hits.
- [ ] Index is cached or rebuilt with clear cost.
- [ ] Regression test proves BM25 recovers a vector-missed document.

## Commit boundary

Use one commit for this issue only:

```text
fix(#10): search BM25 over full corpus
```
