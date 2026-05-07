# easy-local-rag — Agent Entry Point

Working branch: `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`

## Open Work
See [TODO.md](TODO.md) for prioritized issues.
Live query: `gh issue list --repo sriharshaguthikonda/easy-local-rag --state open`

## Source of Truth
- Issues: https://github.com/sriharshaguthikonda/easy-local-rag/issues
- Priority labels: `priority:P0` / `priority:P1` / `priority:P2`
- Type labels: `type:bug` / `type:security` / `type:perf` / `type:enhancement` / `type:todo` / `type:dx`

## Notes
- Issue #2 code-path hardcoded keys were removed in this branch. **History still contains exposed keys**, so credential rotation and history cleanup remain required operational tasks.
- ~7 entry points: CLI `localrag.py`, email RAG, Streamlit `streamlit_app.py`, PyQt `GUI_direct_search.py`, ingestion `Text_embeddings_to_chromadb_python.py`, watcher `monitor_file_changes_update_chromaDB.py`, vault writer.
- Ingest model is hardcoded `mxbai-embed-large` via Ollama; query model must match (see #11) or vector space drifts silently.
- ChromaDB collection metadata only contains `file_name` + `modification_time` — embeddings must be queried via `include=["embeddings"]`, not pulled from metadata (#6).
- Branch name encodes WIP scope: GUI + BM25 hybrid + kokoro TTS + token limit + synonyms + monitor changes + Streamlit + ChromaDB docs. Many half-finished modules; check that `Identify_missing_chunks.py` / `deduplicate_ids_chromadb.py` are not used in the active flow.

## Minimum Validation
- `python -m pytest tests -q`
- `python -m py_compile streamlit_app.py rag_gui.py GUI_direct_search.py`
- `gh issue list --repo sriharshaguthikonda/easy-local-rag --state open`

## Do Not Commit
- `.env` or credential files
- `__pycache__/`, `*.pyc`
- local Chroma data under `chroma/` and generated vault files
