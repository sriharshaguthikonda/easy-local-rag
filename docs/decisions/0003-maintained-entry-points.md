# 0003 — Declare a maintained entry-point set; mark the rest unmaintained

**Date:** 2026-08-02
**Status:** Accepted

## Context

The repository root holds 52 Python files and roughly 13 things that can actually be run. Many are
abandoned experiments, Milvus variants, `*backup*.py` copies, and one-off embedding scripts. Several
carry hardcoded absolute paths to directories that no longer exist
(`C:\Users\deletable\OneDrive\easy-local-rag`, `C:\Users\deletable\OneDrive\Kokoro-82M`).

Every "route the helper into the remaining callers" issue — `#5`, `#11`, `#13`, `#10` — therefore
runs into the same question: route into all of them, or some? Routing a prompt-injection guard into
eight entry points, six of which nobody runs and at least two of which cannot start because their
paths are gone, produces a large diff, a large test surface, and a false impression of coverage.

## Decision

Declare a maintained set. Fixes, guards, contracts and tests apply to it.

**Maintained:** `streamlit_app.py`, `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `rag_gui.py` and the
`GUI_*.py` modules, `groq_lama_chromadb_RAG_ETTS.py`, `context_retrieval.py`,
`monitor_file_changes_update_chromaDB.py`, `Text_embeddings_to_chromadb_python.py`.

**Unmaintained:** `localrag.py`, `localrag_no_rewrite.py`, `emailrag2.py`, `collect_emails.py`, the
`Ollama_*.py` family, the Milvus files (`*MIlvs*`, `*Milvus*`), every `*backup*.py`,
`force_GPU_*`, and the one-off embedding/chunking scripts.

Unmaintained files get a header banner and a list in `AGENTS.md`. They do **not** get fixes, and
their absence from a fix is not a gap.

## Rationale

Honest scope beats broad pretence. A guard applied to a file nobody executes buys no safety and
costs review attention, test surface, and the ability to tell real coverage from theatre. Naming the
maintained set makes each issue's closure gate checkable: "routed everywhere it matters" becomes a
finite, listed claim instead of a judgement call repeated per issue.

## Consequences

- Issue closure evidence cites the maintained set; unmaintained files are explicitly out of scope.
- `#26` (repo cleanup) inherits this list as its deletion/archival candidate set.
- Promoting a file back to maintained means routing every current contract into it first.
- Risk accepted: someone runs an unmaintained script and gets unguarded behaviour. The banner is the
  mitigation; deletion under `#26` is the eventual fix.
