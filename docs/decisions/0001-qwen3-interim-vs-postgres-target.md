# 0001 — qwen3 embeddings are an interim step; pgvector remains the target

**Date:** 2026-08-02
**Status:** Accepted
**Context:** [ISSUE-017](../issues/ISSUE-017-postgres-consolidation.md) · [ISSUE-020](../issues/ISSUE-020-deterministic-ingestion.md) · [proposal](../proposals/qwen3-embedding-migration-implementation.md)

## Context

Two migrations have been planned for the same retrieval layer and they contradict each other.

`ISSUE-017` is the canonical epic: consolidate onto the PostgreSQL/pgvector memory platform.
`ISSUE-020` states that the qwen3 embedding proposal is *"superseded historical context, not
implementation authority."*

Separately, a detailed qwen3 migration plan (`mxbai-embed-large` → `qwen3-embedding:0.6b`, new Chroma
store) was written and committed on a working branch, and work proceeded there for a while on the
assumption it was live. Neither document referenced the other, so a session was spent executing a
plan that had already been demoted — and the contradiction was invisible from the branch where the
code lived.

Both migrations are currently at zero implementation: there is no pgvector schema, driver or
container, and no qwen3 store or config.

## Decision

Land the qwen3 migration now as an explicitly **interim, non-authoritative** step. Keep
PostgreSQL/pgvector (`#17`) as the authoritative long-term target. Neither supersedes the other in
the way `ISSUE-020` implies: `#17` supersedes qwen3 as *authority*, not as *near-term work*.

## Rationale

The qwen3 slice is additive and cheap to unwind. It creates a **new** Chroma path and collection,
leaves the existing `mxbai` collection untouched, and rollback is a three-variable environment
change. It therefore delivers retrieval quality now without pre-empting or complicating the pgvector
cutover, which is gated behind `#2` containment and a long handoff chain and is not close to
starting.

The cost of *not* deciding is proven: one stalled session and a branch that executed an overruled
plan.

## Consequences

- The qwen3 document lives in `docs/proposals/`, not `docs/issues/`. It is not a canonical
  `ISSUE-NNN` acceptance contract and cannot relax one.
- Embedding-model identity must be enforced everywhere first (`#11`), or the two stores silently mix.
- `#19`'s Chroma audit and `#20`'s deterministic ingestion must treat **both** collections as input.
- If pgvector work starts before qwen3 finishes, qwen3 is abandoned rather than completed.

## Alternatives rejected

**Go straight to pgvector.** Correct in principle, but it is gated on `#2` (blocked on a human),
then `#18`, `#19`, `#20` and two cross-repository `.memory` handoffs. Waiting means shipping nothing.

**Drop qwen3 entirely as `ISSUE-020` implies.** Defensible, but discards a complete, specific,
low-risk plan and leaves retrieval on `mxbai` indefinitely for no gain.
