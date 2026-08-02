# Issue #38: Preserve retrieved evidence through the Streamlit model prompt

[Roadmap ledger](README.md)

**Status:** COMPLETED / CLOSED — #37 closed
2026-07-26; implemented 2026-08-02 on
`codex/issue-038-evidence-contract` from frozen base `88d0758`. Packet:
`docs/superpowers/plans/2026-08-02-issue-038-streamlit-evidence-contract-implementation.md`.
[PR #42](https://github.com/sriharshaguthikonda/easy-local-rag/pull/42) merged
with Create a merge commit as `068e0c8f986bf6e292d401d8d97b181815fc0593` from
reviewed head `5e5e8941670f519e6447a770fe0f0d012b96cbcc`.
[Review](https://github.com/sriharshaguthikonda/easy-local-rag/pull/42#issuecomment-5155438619),
[verification](https://github.com/sriharshaguthikonda/easy-local-rag/pull/42#issuecomment-5155439139),
and [closure evidence](https://github.com/sriharshaguthikonda/easy-local-rag/issues/38#issuecomment-5155449007):
focused 19 passed; full 175 passed / 4 accepted frozen GUI failures; compile,
fallback grep, ledger, lineage, and diff gates passed. #2 remains blocked;
#38 does not advance #2/#18/#26.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/38
**Labels / priority:** `priority:P0`, `type:bug`
**Dependencies:** Explicit independent P0 exception after #37 only. #2 stays
blocked and #38 neither unlocks nor bypasses the #2 -> #18 -> #26 sequence.

## Canonical acceptance contract

Freeze a clean worktree at the live GUI branch tip; do not merge `main` into
that GUI lineage. Preserve `get_relevant_context_hybrid(...) -> tuple[str,
list[dict]]`, but make every returned source retain non-empty retrieved
`document`, `file_name`, `score`, and existing safe metadata. Derive the legacy
joined context from that one ordered source list.

`build_numbered_sources()` must preserve source order and text while adding
sequential `citation_id` and `source_name`. `build_guarded_context_block()`
must reject missing/empty text before either provider is called. Remove the
`"Answer this yourself!"` retrieval fallback: retrieval failure visibly
abstains/errors without a model call. Unknown `[N]` citation IDs visibly fail
validation and are never silently accepted.

The same source contract applies to Standard, Focused Search, Brain Dump, and
Summary modes. A later packet must prove: one unique sentinel enters the model
prompt exactly once; two chunks keep text-to-citation mapping; all four modes
share the contract; empty text prevents provider calls; joined context equals
the structured sources; and invented citations fail validation.

## Closure gate

After a decision-complete #38 packet, use pristine/RED/GREEN commits, exact-head
review, independent verification, a frozen-target guard, and Create-a-merge-
commit merging. Then #5 may receive a verification-only audit; #7 and #12 may
receive their specified fresh/recovery smokes. Any audit needing code or new
tests remains open behind #2.
