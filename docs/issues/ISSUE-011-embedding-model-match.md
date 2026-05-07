# Issue #11 Plan: Enforce embedding model match

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/11

Priority: P1 bug

## Goal

Prevent users from querying a Chroma collection with a different embedding model
than the one used to ingest it.

## Files to inspect

- `monitor_file_changes_update_chromaDB.py`
- `Text_embeddings_to_chromadb_python.py`
- `streamlit_groq_lama_chromadb_RAG_ETTS.py`
- `GUI_direct_search.py`
- `GUI_settings.py`
- optional new file: `embedding_contract.py`
- tests under `tests/`

## Implementation steps

1. Add a shared constant:

   ```python
   DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
   ```

2. Add helper functions in `embedding_contract.py`:

   - `get_collection_embedding_model(collection)`
   - `set_collection_embedding_model(collection, model_name)`
   - `assert_embedding_model_matches(collection, requested_model)`

3. Store the model name in collection metadata when creating or first writing to
   a collection:

   ```python
   metadata={"embedding_model": DEFAULT_EMBEDDING_MODEL}
   ```

4. In ingest scripts, before adding embeddings:

   - if collection metadata has no embedding model and count is zero, set it
   - if collection metadata has no embedding model and count is nonzero, stop and
     ask for manual migration
   - if metadata exists and differs, raise a clear error

5. In query paths, call `assert_embedding_model_matches()` before generating the
   query embedding.
6. In `GUI_direct_search.py`, reject `settings["embedding_model"]` if it differs
   from collection metadata.
7. Surface mismatch in the UI as a readable error:

   ```text
   Collection was indexed with mxbai-embed-large, but query uses nomic-embed-text.
   Re-index or switch the query model.
   ```

8. Do not silently update metadata for non-empty collections. That would hide an
   already-mixed vector space.

## Tests and verification

Add tests:

- empty collection with missing metadata can be initialized
- non-empty collection with missing metadata raises manual migration error
- matching model passes
- mismatched model raises clear error

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile embedding_contract.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py streamlit_groq_lama_chromadb_RAG_ETTS.py GUI_direct_search.py
```

Manual smoke:

1. Create collection metadata with `embedding_model=mxbai-embed-large`.
2. Set GUI embedding model to a different model.
3. Run a search.
4. Expected: clear mismatch error, no query sent to Chroma.

## Acceptance checklist

- [ ] Ingest writes embedding model metadata.
- [ ] Query checks metadata before embedding.
- [ ] Mismatch fails loudly.
- [ ] Non-empty legacy collections are not silently relabeled.
- [ ] Tests cover match, mismatch, and legacy metadata.

## Commit boundary

Use one commit for this issue only:

```text
fix(#11): enforce embedding model contract
```
