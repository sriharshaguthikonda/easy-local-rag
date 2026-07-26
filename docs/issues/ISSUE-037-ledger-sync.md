# Issue #37: Keep the canonical ledger synchronized with live GitHub state

[Roadmap ledger](README.md) · [implementation packet](../superpowers/plans/2026-07-27-issue-037-ledger-validator-implementation.md)

**Status:** ACTIVE — this activation PR reconciles the known #8/#15,
branch-inventory, and lifecycle drift. Implementation is gated by this
activation PR's GitHub merge commit recorded in immutable PR evidence.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/37
**Labels / priority:** `priority:P1`, `type:dx`
**Dependencies:** #2 remains `blocked-awaiting-user`. #38 is the only approved
independent P0 exception after #37; it does not unlock or bypass #2 -> #18 ->
#26.

## Maintained contract

Add `scripts/validate_issue_ledger.py`, using Python stdlib and the existing
`gh` CLI only. Its public command is:

```powershell
python scripts/validate_issue_ledger.py --repo OWNER/REPO
```

It defaults to `docs/issues/README.md` and `docs/plans/branch-inventory.md`.
It prints only identifiers and expected/actual lifecycle data: exit `0` means
consistent, `1` means one or more ledger mismatches, and `2` means bad usage,
missing `gh`, authentication, or GitHub API failure. It must never print
credentials or GitHub tokens.

The validator compares the live issue state for every tracked row, enforces at
most one active issue (the declared activated issue is sole active; stable
between-issues/after-closure state may have zero), verifies each referenced PR's open/closed/merged
disposition and requires its merge SHA exactly when it is merged, verifies
branch-tip SHAs whenever a document uses them as an execution gate, and rejects
known stale lifecycle phrases including `future activation merge`. It reads
both the roadmap and every `docs/issues/ISSUE-*.md` canonical file, so stale
status/lifecycle text cannot hide outside the index. It rejects a ledger
missing #37 or #38. Deterministic tests mock the GitHub snapshot; no
network-dependent CI gate is added.

## Required implementation sequence

1. This activation PR records #8/#15 closure, PR #36 merge
   `88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205`, current branch/PR state, #37
   as the sole active issue, #38 queued, and #2 blocked. Its immutable GitHub
   merge evidence is the activation gate; this document intentionally does not
   predict that merge SHA.
2. Create a clean implementation worktree from that fetched, frozen target SHA.
   Record pristine baseline first. The current `main` packet base
   `ca1af851cbb9da99137dc7ff677e06ef60303d84` has no `tests/` directory and no
   `streamlit_app.py`, `rag_gui.py`, or `GUI_direct_search.py`; the recorded
   baseline `python -m pytest tests -q` therefore exits 4 and compilation has
   no targets. Re-measure on the activated target; do not treat this as a pass.
3. Commit a failing mocked-validator test change only as
   `test(#37): specify issue ledger validation`, then production GREEN only as
   `fix(#37): validate canonical issue ledger`. Each accepted review finding is
   one small `fix(#37): address accepted review finding` commit.
4. Require exact-head re-review and an independent verifier. Immediately before
   a GitHub **Create a merge commit**, refetch the frozen target and stop on any
   SHA change. Never squash or rebase away the test, production, or accepted
   finding commits.

## Required test cases and closure gate

Mocked snapshots must fail for: stale issue state, stale merged PR SHA, stale
branch tip, a closed-unmerged PR #1-style disposition, duplicate active issue, missing #37/#38, obsolete activation
wording, and stale lifecycle/status wording in a named canonical
`ISSUE-015-*.md` fixture. Reconciled fixtures must pass for both zero-active
between-issues state and one declared active issue. Run focused
tests, affected-file compilation, `git diff --check`, and the full suite,
comparing full-suite failures with a freshly recorded pristine baseline.

The later implementation-owned `AGENTS.md` edit must name the root orchestrator
as owner: immediately after every activation, implementation merge, issue
closure, branch archive, or PR disposition change, the root orchestrator
updates the ledger and runs the validator before any next activation or
closure. The command is mandatory immediately before every activation and
closure. After implementation merge, post SHA-bound review and verifier
evidence, close #37 with `COMPLETED`, make a small closure-ledger PR, and run
the live validator again. Roll back the implementation with
`git revert -m 1 <merge-sha>`; before merge, revert child commits in reverse
order. No issue is closed without packet, implementation PR/commits, review,
verifier, merge, rollback, and final-ledger evidence.
