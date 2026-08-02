# Issue #11: Enforce the embedding-model contract

[Roadmap ledger](README.md)

**Status:** OPEN — `embedding_contract.py` and GUI checks exist; complete ingest/query coverage and legacy-collection tests are required.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/11
**Labels / priority:** `priority:P1`, `type:bug`
**Dependencies:** Carry model provenance through #19/#20 and test the maintained legacy caller; otherwise close from retirement proof. #7 creation and #10 retrieval use this contract only while supported.

## Implementation slices

1. Set `embedding_model` metadata when creating or first writing an empty collection.
2. Reject a non-empty collection without metadata and every requested-model mismatch before embedding/querying.
3. Route monitor, ingest, Streamlit, and GUI paths through the shared helper; surface a concise re-index/switch-model error.

## Affected interfaces, files, and artifacts

- `embedding_contract.py`, ingestion/monitor scripts, Streamlit backend, `GUI_direct_search.py`, `GUI_settings.py`, tests.
- Chroma collection metadata: `embedding_model` is authoritative; dimensions alone do not prove compatibility.

## Concrete actions

- Do not relabel a non-empty legacy collection; require an explicit migration/re-index.
- Audit every caller generating embeddings, not only GUI direct search.
- Keep the default model in one shared location where practical.

## Verification

```powershell
python -m pytest tests/test_embedding_contract.py -q
python -m pytest tests -q
python -m py_compile embedding_contract.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py streamlit_groq_lama_chromadb_RAG_ETTS.py GUI_direct_search.py
```

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** empty initialization works, matching models pass, mismatch and non-empty missing metadata fail before querying, and all maintained ingest/query paths use the helper.
- **Retirement closure (mutually exclusive):** remove uncovered ingest/query paths from supported entry points and docs, prove they cannot generate or query embeddings, and link a maintained contract-enforcing replacement. Every remaining path must still reject mismatches.
- **Rollback constraint:** never silently alter non-empty metadata to bypass an error.
- **Commit:** `fix(#11): enforce embedding model contract`.
