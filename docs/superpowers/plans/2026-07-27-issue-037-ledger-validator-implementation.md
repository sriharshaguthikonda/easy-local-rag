# Issue #37 Ledger Validator Implementation Packet

[GitHub issue #37](https://github.com/sriharshaguthikonda/easy-local-rag/issues/37) · [canonical plan](../../issues/ISSUE-037-ledger-sync.md) · [roadmap ledger](../../issues/README.md)

**Packet base:** `origin/main` at `ca1af851cbb9da99137dc7ff677e06ef60303d84`.
This packet is immutable implementation detail: it cannot activate #37, repair
the already-known #8/#15 drift, or weaken #2's blocking sequence.

## Locked scope

Add only `scripts/validate_issue_ledger.py`, focused deterministic tests, and
the mandatory command wording in `AGENTS.md` during the later implementation
PR. Use only Python stdlib and existing `gh`; no dependency, CI network gate,
or GitHub mutation is allowed. The CLI accepts required `--repo OWNER/REPO`
and optional document-path overrides, defaults to `docs/issues/README.md` and
`docs/plans/branch-inventory.md`, and returns: `0` consistent, `1` mismatch,
`2` usage/API/CLI/auth failure. Output names identifiers and expected/actual
state only; never secrets.

Parse the canonical tracked-issue rows and lifecycle text from both
`docs/issues/README.md` and every `docs/issues/ISSUE-*.md`, obtain one live
snapshot through `gh` JSON output, and compare: issue open/closed state; the
single-active invariant; every referenced merged PR SHA; every branch tip used
as an execution gate; presence of #37/#38; and stale phrases such as `future
activation merge`. Keep parsing intentionally narrow to the documented ledger
forms; an unparseable required row is a mismatch, not a false pass.

## Frozen activation and ownership

The separate activation PR is the first code gate. It owns only the canonical
ledger/plan rows necessary to record #8 and #15 closed, PR #36 merge
`88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205`, refreshed main/GUI/PR #1 state,
removal of obsolete future-activation wording, #37 as sole active, #38 queued,
and #2 blocked. Fetch and freeze that target SHA before implementation; stop if
it changes before work or merge. Do not merge `main` into the later #38 GUI
lineage.

The implementation branch owns `scripts/validate_issue_ledger.py`, its focused
tests, and the required `AGENTS.md` command. It must not alter #38 code,
credential history, generated data, or unrelated ledger state.

## TDD and evidence

Record a pristine baseline on the activated target before edits. The packet
base itself has no `tests/` directory and no named GUI compile targets:
`python -m pytest tests -q` exited 4 with `file or directory not found: tests`,
while `git diff --check` exited 0. Re-run and record this exact comparison on
the activated target.

First add deterministic mocked-snapshot tests and record RED for stale issue,
stale PR SHA, stale branch tip, duplicate active, absent #37/#38, obsolete
activation wording, and stale status/lifecycle wording in a named canonical
`ISSUE-015-*.md` fixture; commit only those tests as:

```text
test(#37): specify issue ledger validation
```

Implement the minimal parser/CLI and record GREEN for each failing case plus a
reconciled fixture; commit only production as:

```text
fix(#37): validate canonical issue ledger
```

Run focused tests, affected-file compilation, `git diff --check`, and full
suite comparison against pristine. Each accepted reviewer finding receives one
small commit; rejected findings get a written disposition. The reviewer and
independent verifier must PASS the same final SHA.

## Merge, rollback, and closure

Immediately before merge, refetch the implementation target and confirm the PR
base/head match frozen evidence. Merge with GitHub **Create a merge commit**;
never squash or rebase. Roll back after merge with `git revert -m 1 <merge-sha>`
or, before merge, child commits in reverse order. Post the immutable packet,
review, verifier, test, PR, commit, merge, rollback, and live-validator
evidence; close #37 as `COMPLETED`; merge a narrow closure-ledger PR; rerun the
live validator. Only then may #38 receive its own JIT packet.
