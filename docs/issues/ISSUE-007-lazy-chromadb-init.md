# Issue #7: Lazy-initialize Chroma for fresh installs

**Status:** OPEN — live code has lazy call sites, but initialization semantics and fresh-path proof remain incomplete.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/7
**Labels / priority:** `priority:P1`, `type:bug`
**Dependencies:** #11 collection metadata must be set correctly when a collection is first created.

## Implementation slices

1. Remove import-time collection opening; `initialize_collection` accepts a client/name and creates only at runtime.
2. Use `get_or_create_collection` for the onboarding creation path; expose a readable error/guidance path when creation is intentionally disabled.
3. Cache one client/collection resource per Streamlit session and test fresh startup with a fake client and empty temporary path.

## Affected interfaces, files, and artifacts

- `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `streamlit_app.py`, `GUI_chromadb.py`, tests.
- Public behavior: importing Streamlit code does not require a preexisting Chroma collection.

## Concrete actions

- Audit callers of `initialize_collection`; do not create clients per rerun.
- Surface onboarding rather than an import traceback when the collection is absent.
- Confirm existing collections still load without unintended recreation.

## Verification

```powershell
python -m pytest tests -q
python -m py_compile streamlit_groq_lama_chromadb_RAG_ETTS.py streamlit_app.py
streamlit run streamlit_app.py
```

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** module import makes no Chroma call; an empty temporary Chroma path launches with setup guidance or a new collection; an existing collection still loads.
- **Retirement closure (mutually exclusive):** remove an affected entry point from supported docs and launch surfaces, prove importing every maintained entry point makes no eager Chroma call, and link its maintained lazy-initialized replacement.
- **Rollback constraint:** do not reintroduce import-time `get_collection`; retain data and metadata on any creation-path rollback.
- **Commit:** `fix(#7): lazy-init Chroma collection`.
