# Issue #11: Enforce the embedding-model contract

[Roadmap ledger](README.md)

**Status:** OPEN — active under the one-time maintained Chroma implementation exception after the separate activation merge. Shared coverage and CSV provenance are required.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/11
**Labels / priority:** `priority:P1`, `type:bug`
**Dependencies:** The [owner approved the #11-only exception](https://github.com/sriharshaguthikonda/easy-local-rag/issues/11#issuecomment-7191003443) while #2 remains open. After the separate activation merge, implement and independently verify the maintained Chroma readers and writers, including imported CSV provenance. #10 stays blocked until independent verification passes. #19/#20 migration provenance and all #2 closure criteria stay gated.

[Implementation packet](../superpowers/plans/2026-11-26-issue-011-chroma-embedding-contract-implementation.md)

## Implementation slices

1. Set `embedding_model` at creation, preserve existing collection metadata, and reject writer access to any existing unlabelled collection.
2. Reject unreadable or invalid counts when empty state must be proved. Reject unknown non-empty and mismatched collections before embedding, querying or writing. Return empty for a proved-empty unlabelled reader.
3. Route every maintained vector caller through the shared guard. Require digest-bound model provenance and fully validate the CSV snapshot before Chroma initialization or writes.

## Affected interfaces, files, and artifacts

- `embedding_contract.py`, ingestion/monitor scripts, Streamlit backend, `GUI_direct_search.py`, `GUI_settings.py`, tests.
- Chroma collection metadata: `embedding_model` is authoritative; dimensions alone do not prove compatibility.

## Concrete actions

- Preserve all existing collection labels. Required recovery for an unlabelled destination is a separately authorized rebuild or a new correctly labelled collection.
- Audit every caller generating embeddings, not only GUI direct search.
- Keep the default model in one shared location where practical.

## Verification

```powershell
python -m pytest tests/test_embedding_contract.py -q
python -m pytest tests -q
python -m py_compile embedding_contract.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py streamlit_groq_lama_chromadb_RAG_ETTS.py GUI_direct_search.py
```

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** new labelled creation and matching models pass. Unreadable counts, incompatible models, unknown provenance and invalid CSVs fail before their protected operations. All maintained ingest/query paths use the guard. Independent verification on synthetic Chroma and immutable merged evidence are required. This does not satisfy the #19/#20 handoff.
- **Retirement closure (mutually exclusive):** remove uncovered ingest/query paths from supported entry points and docs, prove they cannot generate or query embeddings, and link a maintained contract-enforcing replacement. Every remaining path must still reject mismatches.
- **Rollback constraint:** never silently alter non-empty metadata to bypass an error.
- **Commit:** `fix(#11): enforce embedding model contract`.
