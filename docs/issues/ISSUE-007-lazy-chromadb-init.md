# Issue #7 Plan: Lazy-init ChromaDB collection for fresh users

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/7

Priority: P1 bug

## Goal

A fresh clone with an empty Chroma directory must be able to launch the app.
Importing `streamlit_app.py` must not crash just because the collection does not
exist yet.

## Files to inspect

- `streamlit_groq_lama_chromadb_RAG_ETTS.py`
- `streamlit_app.py`
- `GUI_chromadb.py`
- tests under `tests/`

## Implementation steps

1. In `streamlit_groq_lama_chromadb_RAG_ETTS.py`, remove import-time collection
   initialization:

   ```python
   if collection is None:
       collection = initialize_collection()
   ```

2. Change `initialize_collection()` to use `get_or_create_collection()` instead
   of `get_collection()`.
3. Add optional arguments to `initialize_collection()`:

   - `name=collection_name`
   - `create=True`
   - `client=None`

4. If `create=False`, return a clear error when the collection is missing. If
   `create=True`, create it.
5. In `streamlit_app.py`, call `initialize_collection()` inside session setup and
   catch expected Chroma errors. Show an onboarding message instead of crashing.
6. Ensure `get_relevant_context_hybrid()` initializes lazily if `collection is
   None`.
7. Avoid creating multiple clients per rerun. Store the collection or client in
   Streamlit session state or a cached resource.

## Tests and verification

Add tests with a fake Chroma client:

- `initialize_collection(create=True)` calls `get_or_create_collection`
- importing the module does not call Chroma
- a missing collection returns a UI-visible error, not an import exception

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile streamlit_groq_lama_chromadb_RAG_ETTS.py streamlit_app.py
```

Manual smoke:

1. Temporarily point Chroma to an empty temp directory.
2. Run:

   ```powershell
   streamlit run streamlit_app.py
   ```

3. Expected: app loads and either creates the collection or shows setup guidance.
4. Expected: no import-time `Collection ... does not exist` crash.

## Acceptance checklist

- [ ] No Chroma collection is opened at import time.
- [ ] Fresh Chroma path can launch the UI.
- [ ] Existing Chroma path still loads existing collection.
- [ ] Missing/empty collection has clear UI guidance.
- [ ] Tests cover fresh-user startup.

## Commit boundary

Use one commit for this issue only:

```text
fix(#7): lazy-init Chroma collection
```
