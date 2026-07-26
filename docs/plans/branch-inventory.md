# Public branch and PR inventory

Snapshot date: 2026-07-27. This is a dated, non-authoritative public GitHub
inventory required by issue #18. It intentionally omits credential values and
private corpus content. It is orientation only: immediately before any
execution gate, activation, closure, branch archive, or PR disposition, fetch
the relevant ref and run `python scripts/validate_issue_ledger.py --repo
OWNER/REPO`; do not use this snapshot as a branch-tip execution gate.

## `main`

- Tip at packet-merge snapshot: `f538d33a29bbc233cceee30444f42a3bf43b9466`
- Role: old upstream-derived CLI and email RAG code; this is the current default branch, not the advanced GUI implementation.
- Current problem: README and layout still describe the old Ollama/Chroma workflow and do not represent the PostgreSQL/pgvector target.
- Disposition: retain as the migration base. Add new PostgreSQL work in focused branches; do not replace it with an experimental branch merge.

## `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`

- Tip at snapshot: `88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205`
- Relation to `main`: verify live before execution; this dated snapshot is not
  a frozen implementation target.
- Main additions: several PyQt modules, `rag_gui.py`, `streamlit_app.py`, a large Chroma/Groq/Ollama application module, direct-search and context helpers, file monitoring, semantic chunking, Chroma utilities, TTS code, and GUI tests.
- Executable/client candidates: `rag_gui.py`, `streamlit_app.py`, `groq_lama_chromadb_RAG_ETTS.py`, the monitor batch file, and ingestion/query utility scripts.
- Database assumptions: direct ChromaDB access from clients and workers; multiple hard-coded or historical collection names, including `html_chunks_temp` and `html_chunks_text_in_documents`; no shared PostgreSQL service boundary.
- Provider assumptions: Ollama plus optional/embedded Groq paths, Google TTS and speech-recognition code. Local and cloud behaviour is mixed rather than expressed as explicit modes.
- Useful salvage:
  - background worker and progress/error patterns;
  - streaming result presentation;
  - search-only interaction;
  - source/context panes and score display;
  - database status/reconnect ideas;
  - deterministic chunk-ID and file-change concepts;
  - tests that can be converted into contract or regression fixtures.
- Known broken or unsafe areas:
  - legacy retrieval defects tracked by #5–#14;
  - BM25 is derived from vector candidates rather than an independent corpus search (#10);
  - collection access and provider logic are mixed into clients;
  - personal Windows paths and runtime downloads occur at import/startup;
  - broad exception handling hides retrieval failures;
  - generated `__pycache__`, `.pyc`, logs, caches and browser assets are committed.
- Disposition: **archive and mine selectively**. Do not merge wholesale. Port only behaviour that has a destination contract and tests under #20–#24 and #27.

## `codex/add-qtpy-gui-for-groq_lama_chromadb_rag_etts.py`

- Tip: `8cb08091347ee41e7c8dac1c04d02a8e5d2eed2b`
- Relation to `main`: 16 commits ahead, 0 behind.
- Relation to the current BM25 branch: diverged; 1 commit ahead and 6 commits behind, with merge base `b293135ff91befb0c27d9725b5e7001ff9cc1d0f`.
- Unique change: a 98-line QtPy wrapper plus one README line and one dependency line.
- Database/provider assumptions: inherits the older experimental Chroma/Groq application directly rather than using a shared search/chat boundary.
- Useful salvage: minimal Qt wrapper shape only; the future GUI requirements are already captured by #24.
- Disposition: **close PR #1 as superseded and archive the SHA**. Do not rebase or merge this branch.

## PR #1 — `Add QtPy GUI for Groq chat`

- State at inspection: **CLOSED, unmerged**.
- Base: the experimental BM25 branch, not `main`.
- Head: `8cb08091347ee41e7c8dac1c04d02a8e5d2eed2b` on the stale/diverged QtPy
  branch above.
- Decision: superseded by #17 and #24. Its useful idea is recorded as a future thin-client option; its architecture is not a merge path.

## PR #36 — Atomic vault write

- State at inspection: **MERGED** on 2026-07-24.
- Base: `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
- Reviewed code head: `3ced61bcb9957907cc568e2e2560edd6f3c53b83`.
- GitHub Create-a-merge-commit SHA:
  `88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205`.

## Closed fixes not present on `main`

Issues #3 and #4 were closed against commits `4686575a55358a0d5bf74447dc5e083e1361f66b` and `07b546964588cf9bce85ec35d9775a58b6266477`. Both commits are descendants of the experimental GUI history and are not part of `main` or either named branch tip inspected above.

Treat these commits as salvage evidence, not as proof that the default branch or future PostgreSQL client is fixed. If file opening or subprocess search is reintroduced, write new tests against the new service/client boundary and port the safety properties rather than cherry-picking the old application files.

## Archive and cleanup rules

1. Rotate/revoke exposed credentials before preserving any public archive ref.
2. Run a secret scan across all remote, tag, pull-request and local refs; publish only secret types/locations, never values.
3. Record each local-only branch name and tip SHA in this document before deletion.
4. Create immutable sanitized archive refs only after the scan.
5. Close PR #1 after this disposition is linked from GitHub.
6. Delete no branch until #19 export evidence and #25 rollback gates exist.
