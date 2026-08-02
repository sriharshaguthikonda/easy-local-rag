# Roadmap

The authoritative status, dependencies, closure gates and execution order live in the
[canonical issue-plan index](issues/README.md). This file is the human entry point: it says where
things stand and where to look next. When the two disagree, the ledger wins.

Authority precedence: **GitHub issue = live state and human decisions · canonical plan = scope,
invariants, closure gates · roadmap ledger = ordering and handoffs · JIT packet = implementation
detail, and cannot weaken a canonical plan.**

## Where the work lives

All implementation is on the GUI lineage, branched from `88d0758` (the tip of
`GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`, PR #36). The canonical planning
documents were brought onto that lineage with `git checkout origin/main -- <paths>` rather than a
merge, because [ISSUE-038](issues/ISSUE-038-streamlit-evidence-contract.md) forbids merging `main`
into this lineage. See [ADR 0005](decisions/0005-single-lineage-consolidation.md).

## Two tracks

**Track A — legacy hardening.** Finish the half-routed fixes in the existing app. Each of these has
a helper module that was written and wired into exactly one caller, leaving the other callers on the
original broken code.

**Track B — consolidation epic (#17).** Move the app onto the PostgreSQL/pgvector memory platform:
`#18` inventory → `#19` Chroma audit/export → `#20` deterministic ingestion → `#21`/`#22` thin
client and provider modes → `#23` retrieval parity → `#27` grounded answers → `#25` Chroma
retirement → `#26` repo cleanup. **Track B is gated on #2** and has not started.

## Status

| # | Title | Track | Status |
|---|---|---|---|
| 2 | Rotate/remove hardcoded Groq keys | A | **blocked — needs the maintainer.** Code paths clean; git history still contains keys. Rotation and history remediation are human actions. Gates all of Track B. |
| 5 | Prompt injection from retrieved documents | A | partial — guard routed into 2 of the maintained callers |
| 6 | MMR embedding KeyError | A | closed (PR #29) |
| 7 | Lazy Chroma init | A | present for the Streamlit entry point; verification only |
| 8 | Config-driven paths | A | closed |
| 9 | Validate conversation import | A | closed (PR #33) |
| 10 | Full-corpus BM25 | A | partial — fixed in `GUI_direct_search.py` only; the live `rag_gui.py` path still uses top-N |
| 11 | Embedding model match | A | partial — duplicate shadowing `init_chromadb` leaves ingest unchecked |
| 12 | Streamlit TTS worker | A | present; verification only |
| 13 | tiktoken token counter | A | partial — regex counter still live in two backends |
| 14 | Inline citations | A | closed as duplicate of #38 |
| 15 | Atomic vault write | A | closed (PR #36) |
| 16 | Reproducible setup, pins, entry points | A | partial — zero version pins, no lockfile, no CI |
| 17 | PostgreSQL/pgvector consolidation | B | epic, not started |
| 18–27 | Consolidation packets | B | queued behind #2 |
| 37 | Ledger synchronization | A | closed (PR #41, `f87b4cf`) |
| 38 | Streamlit evidence contract | A | closed (PR #42, Create a merge commit `068e0c8f986bf6e292d401d8d97b181815fc0593`; reviewed head `5e5e8941670f519e6447a770fe0f0d012b96cbcc`) |

An interim [qwen3 embedding migration](proposals/qwen3-embedding-migration-implementation.md) runs
alongside Track A. It is additive and env-reversible and does **not** pre-empt #17 — see
[ADR 0001](decisions/0001-qwen3-interim-vs-postgres-target.md).

## Decisions

- [0001 — qwen3 as interim, pgvector as target](decisions/0001-qwen3-interim-vs-postgres-target.md)
- [0002 — retrieved-source contract](decisions/0002-retrieved-source-contract.md)
- [0003 — maintained vs unmaintained entry points](decisions/0003-maintained-entry-points.md)
- [0004 — deliberate deviation from the #2 gate](decisions/0004-legacy-work-under-issue-2-gate.md)
- [0005 — single-lineage consolidation](decisions/0005-single-lineage-consolidation.md)

## Standing rule: no source-string tests

Several issues were "covered" by tests shaped `assert "..." in Path("file.py").read_text()`. Those
tests pass while the code beneath them is broken — which is exactly how #38 shipped a RAG app that
sent the model no evidence at all. Every issue closed from 2026-08-02 onward must land a behavioural
test, and must replace any source-string test covering the same ground.
