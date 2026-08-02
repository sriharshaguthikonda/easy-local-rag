# 0004 — Legacy hardening proceeds under the #2 gate; consolidation does not

**Date:** 2026-08-02
**Status:** Accepted
**Context:** [ledger](../issues/README.md) · [ISSUE-002](../issues/ISSUE-002-rotate-groq-keys.md)

## Context

The roadmap ledger fixes the execution order as
`#37 → #38 → verification-only #5/#7/#12 audits → #2A → #18 → #26A → #26B → #2 closure → …`
and states: *"Any audit needing code or new tests remains open behind #2."*

`#2` is `blocked-awaiting-user`. It requires rotating Groq API keys that are present in git history
and recording history-remediation evidence. Both are human actions: an agent must not rotate
credentials, and history rewriting needs explicit maintainer approval.

Read literally, the gate parks **all** remaining code work — including five legacy issues that are
demonstrably half-finished — behind an action that may not happen for days. Meanwhile the repository
carries live defects: a duplicate `init_chromadb` that lets ingest write the wrong embedding model's
vectors into a collection without error, and retrieved documents reaching prompts unguarded on
several paths.

## Decision

Split the gate by what it actually protects.

**Proceeding:** legacy hardening — `#11`, `#5`, `#13`, `#10`, `#7`, `#12`, `#16` — plus the interim
qwen3 migration. None of these touch credentials, git history, or the consolidation sequence.

**Not proceeding:** `#2` closure itself, and the entire `#18`–`#27` consolidation epic. These remain
behind the gate exactly as written.

## Rationale

The `#2 → #18 → #26` chain exists so that no branch inventory, history rewrite, or repository
cleanup is performed before exposed credentials are contained — otherwise cleanup work is invalidated
by a later history rewrite, and secrets survive in refs nobody inventoried. Legacy code fixes in the
application layer do not interact with that chain: they do not create refs, do not rewrite history,
and are not invalidated by a later rewrite.

So the deviation is in the letter of the gate, not its substance. Recording it as a decision keeps it
visible and reversible rather than letting it become undocumented drift.

## Consequences

- The maintainer's #2 action list must stay prominent; nothing here reduces its urgency, and the
  security exposure is unchanged.
- Any legacy work that turns out to need a branch, a ref rewrite, or history access **stops** and
  parks behind #2.
- If the maintainer disagrees, the legacy commits are independent and can be parked without
  disturbing Track B, which has not started.

## Alternatives rejected

**Follow the gate literally and stop.** Honest to the ledger, but leaves known data-corruption and
prompt-injection defects live for an unbounded wait on a human action, and the maintainer explicitly
asked for the issues to be worked through.

**Declare #2 closed on the basis that code paths are clean.** Rejected outright. The keys are still
in history; `docs/issues/README.md` is explicit that no containment closure is inferred from docs.
