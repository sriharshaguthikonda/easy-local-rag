# Issue #6 Plan: Fix hybrid retrieval MMR embedding KeyError

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/6

Priority: P0 bug

## Goal

Stop the Streamlit hybrid retrieval path from reading embeddings from metadata.
Chroma metadata only stores file information, while embeddings must come from
the query result's `embeddings` array.

## Files to inspect

- `streamlit_groq_lama_chromadb_RAG_ETTS.py`
- `Text_embeddings_to_chromadb_python.py`
- `monitor_file_changes_update_chromaDB.py`
- tests under `tests/`

## Current bug shape

The retrieval query must include embeddings:

```python
include=["documents", "metadatas", "distances", "embeddings"]
```

But the MMR loop must then read:

```python
res["embedding"]
selected["embedding"]
```

It must not read:

```python
res["meta"]["embedding"]
selected["meta"]["embedding"]
```

## Implementation steps

1. In `get_relevant_context_hybrid()`, keep or add `embeddings` in the Chroma
   `collection.query(...)` include list.
2. Build `vector_results` with these keys:

   - `meta`
   - `document`
   - `embedding`
   - `vector_score`

3. Build keyword results with `embedding=None` unless you can join them back to
   the vector result by stable chunk ID.
4. When combining results, keep the embedding at the top level:

   ```python
   combined_results[file_name] = {
       "meta": res["meta"],
       "embedding": res["embedding"],
       "final_score": ...,
   }
   ```

5. In the MMR loop, compute similarity only when both candidate and selected
   items have embeddings:

   ```python
   if res.get("embedding") is None:
       continue
   np.dot(res["embedding"], selected["embedding"])
   ```

6. Do not swallow this bug with the existing broad `except`. Either re-raise in
   debug mode or return a structured empty result that the UI can show.
7. Keep return shape stable. If callers expect `(context, metadata)`, every
   success and failure path must return a 2-tuple.

## Tests and verification

Add a unit test with a fake collection query result:

- metadatas contain only `file_name`, `modification_time`, and `text`
- embeddings are present in `search_result["embeddings"][0]`
- `additional_unique_files > 0`
- expected: no `KeyError`, no `"Answer this yourself!"`, returns selected docs

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile streamlit_groq_lama_chromadb_RAG_ETTS.py
```

Manual smoke:

1. Run Streamlit against a small Chroma collection.
2. Ask a query with `additional_unique_files` enabled.
3. Confirm the terminal does not print `An error occurred: 'embedding'`.
4. Confirm the UI receives source metadata.

## Acceptance checklist

- [ ] Chroma query includes embeddings.
- [ ] MMR reads top-level result embeddings, not metadata embeddings.
- [ ] Keyword-only results cannot crash MMR.
- [ ] Failure path returns a stable type.
- [ ] Regression test covers metadata without `embedding`.

## Commit boundary

Use one commit for this issue only:

```text
fix(#6): use Chroma embeddings in MMR
```
