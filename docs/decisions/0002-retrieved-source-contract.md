# 0002 — One retrieved-source contract; joined context is derived, never parallel

**Date:** 2026-08-02
**Status:** Accepted
**Context:** [ISSUE-038](../issues/ISSUE-038-streamlit-evidence-contract.md) · [packet](../superpowers/plans/2026-08-02-issue-038-streamlit-evidence-contract-implementation.md) · PR #42

## Context

The Streamlit path retrieved chunk text correctly and then dropped it before building the prompt.
`get_relevant_context_hybrid` returned `final_results = [res["meta"] for res in selected_results]`,
which is metadata only. Downstream, `build_numbered_sources()` had no text to carry, and
`build_guarded_context_block()` fenced empty blocks.

The result was an application that looked like RAG, displayed citations, and sent the model **no
evidence at all** — answering from model knowledge while appearing evidence-backed. On an older
lineage the same area was worse: documents were never zipped in, producing `KeyError`s that a broad
`except` swallowed into the string `"Answer this yourself!"`.

Two properties made this survive. Evidence travelled as **two parallel values** — a joined context
string and a source list — that could silently disagree. And the tests were source-string greps, so
nothing could observe the defect.

## Decision

There is exactly one evidence path. Retrieval returns ordered source objects carrying non-empty
`document` alongside `file_name`, `score` and safe metadata. The joined context string is **derived
from that same list**:

```python
final_results = [{**res["meta"], "document": res["document"], ...} for res in selected_results]
relevant_context = "\n\n".join(source["document"] for source in final_results)
return relevant_context, final_results
```

Empty or missing document text is rejected **before** any provider call. Retrieval failure abstains
visibly or raises — it never returns a plausible-looking non-answer. Unknown `[N]` citation IDs fail
validation. The same contract applies to Standard, Focused Search, Brain Dump and Summary modes.

## Rationale

Deriving the string from the list makes divergence structurally impossible rather than a thing to
remember. Removing the silent fallback converts a class of failures from invisible to loud: that
single string is what hid this bug, and it existed at four sites.

## Consequences

- `#27` (grounded answers) takes this contract as its input dependency.
- Any new retrieval mode must return source objects; returning metadata-only is a contract breach.
- A sentinel test is mandatory: a unique sentence in a fake retrieval result must appear in the
  outgoing model prompt exactly once. This is the regression guard for the whole class.
- The `"Answer this yourself!"` fallback is deleted repository-wide and must not return.
