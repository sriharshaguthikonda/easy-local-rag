# 0005 — One lineage: GUI tip for code, path-checkout for plans

**Date:** 2026-08-02
**Status:** Accepted

## Context

Work had forked into two branches that each held half of what was needed.

`codex/qwen-embedding-migration-plan` carried thirteen fix commits and, after a merge of `main`, all
the canonical planning documents — but its code was an **older lineage** that had never received the
GUI branch's work. It had 80 tests.

The GUI branch tip `88d0758` (PR #36) carried the newer code and 140 tests, but none of the planning
documents, no roadmap ledger, and no `scripts/validate_issue_ledger.py`.

`ISSUE-038` names `88d0758` as its audited code line, and the older lineage had *regressed* the
document-carrying retrieval fix present there. So implementation had to happen on the GUI lineage.
But `ISSUE-038` also states: *"Freeze a clean worktree at the live GUI branch tip; do not merge
`main` into that GUI lineage."*

That is a genuine bind: the plans were only reachable through `main`, and `main` was forbidden.

## Decision

Branch from `88d0758` for code. Bring the planning documents across by **path checkout**, not merge:

```
git checkout origin/main -- docs/issues docs/plans scripts/validate_issue_ledger.py \
                            tests/test_validate_issue_ledger.py
```

## Rationale

A path checkout imports file contents without importing history or code lineage. The contract's
prohibition is about not dragging `main`'s code state into the GUI lineage; copying documentation and
one validator script does neither. A `git merge` would have replayed 57 commits of older code over
newer code — the exact mistake that produced the fork.

## Consequences

- One branch now holds code, plans, ledger and validator. `python scripts/validate_issue_ledger.py`
  runs where the code is, so ledger drift is caught by the normal test run.
- The doc files carry no merge ancestry to `main`. Future doc syncs use the same path-checkout, and
  divergence must be watched for.
- `codex/qwen-embedding-migration-plan` is superseded for code purposes. Its unique value — the
  merged plans and ledger corrections at `64c8b14`/`c371c2c` — has been re-landed here.
- Anything genuinely needed from the old lineage must be cherry-picked deliberately, not merged.
