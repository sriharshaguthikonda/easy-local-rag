# Canonical issue-plan index

This index is the authoritative issue/plan/status index and roadmap ledger for
the 24 tracked issues:
[#2](https://github.com/sriharshaguthikonda/easy-local-rag/issues/2) and
[#5](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5) through
[#27](https://github.com/sriharshaguthikonda/easy-local-rag/issues/27).
Canonical issue files are acceptance contracts, not implementation evidence.
Authority precedence is: **GitHub issue = live state and human decisions;
canonical plan = scope, invariants, and closure gates; roadmap ledger =
ordering and handoffs; JIT packet = implementation detail and cannot weaken a
canonical plan.** The status column reflects live state checked on 2026-07-18:
#14 is closed, while #6 has a merged GUI-lineage fix and remains open only until this canonical plan lands and receives reciprocal evidence links.

## Required execution order

`#2A -> #18 -> #26A -> separately approved #26B -> #2 closure -> #19A -> #19B -> #20A -> .memory #4 handoff -> #20B -> .memory #5 handoff -> #21A -> #22A -> #23 -> #21B -> #22B -> #24 specification -> #27 -> #25 -> GUI/no-GUI handoff -> #26C -> #26D`

#24 is a specification-only issue. It records the future GUI contract and
salvage decisions; it does not authorize GUI implementation. #17 is an epic,
not a packet. #6 is evidence-link closure only, #14 is a closed duplicate, and
neither has a packet. Legacy issues #5-#16 retain their own closure gates but
must not bypass the consolidation sequence.

## JIT packet policy

Only the next unblocked packet may be created, at
`docs/superpowers/plans/YYYY-MM-DD-issue-NNN-<slug>-implementation.md`.
Each packet names one canonical issue/slice and its canonical plan links to it
once created; this ledger tracks its state. The only states are `queued`,
`planning`, `active`, `review`, `blocked-awaiting-user`, and `closed`. Exactly
one issue may be `active` at a time (zero only between packets or while
approvals block execution). A blocked-awaiting-user #2 parks #2; execute #9,
then #15, one at a time, without advancing #18 or #26. If both close and #2
remains blocked, only read-only JIT packet preparation may continue; do not
bypass the #2/#18/#26 security sequence.

Each packet must contain issue/canonical/roadmap links; exact target branch and
frozen base SHA; maintained-path or retirement mode chosen first; current
caller map/files owned; locked interfaces, schemas, I/O, limits, failure, and
rollback behavior; TDD exact commands with expected failure/pass evidence;
commit boundaries, approval stops, and closure evidence. It may contain no
unresolved placeholder, implementer choice, or speculative dependency.

Workers use this lifecycle: `planner agent -> ChatGPT -> GSD checker ->
corrector agent -> implementation agent -> code-review agent -> fixer agent ->
verifier -> orchestrator`. A packet may
describe implementation detail only; it cannot relax a canonical scope,
invariant, closure gate, approval, or rollback constraint.

## Gate ledger and reciprocal handoffs

| Gate producer | Pinned SHA / schema | Required evidence | Consumer | State |
|---|---|---|---|---|
| #2A containment | Not yet produced | revocation, active-tree scan, missing-key behavior | #18 | Not passed; docs are not proof. |
| #18 inventory/proposed disposition | Not yet produced | immutable ref manifest, redacted scan, proposed dispositions | #26A/#26B, #19A | Not passed; docs are not proof. |
| #26B separately approved rewrite | Not yet produced | all-ref post-rewrite evidence or documented, explicitly accepted immutable-ref residual outcome | #2 closure, #19A | Not passed; docs are not proof. |
| #19A audit / #19B export | Not yet produced | frozen snapshot, manifest, checksums, read-only proof | #20A, #25 | Not passed; docs are not proof. |
| #20A | Not yet produced | deterministic package and identity verification | `.memory` #4 | Not passed; docs are not proof. |
| `.memory` #4 | Not yet produced | reciprocal SHA, schema, package version, compatibility command/result | #20B | Not passed; docs are not proof. |
| #20B | Not yet produced | idempotent import and reconciliation evidence | `.memory` #5 | Not passed; docs are not proof. |
| `.memory` #5 | Not yet produced | reciprocal SHA, schema, package version, compatibility command/result | #21A, #22A, #23 | Not passed; docs are not proof. |
| #21A / #22A | Not yet produced | pinned interface compatibility evidence | #23 | Not passed; docs are not proof. |
| #23 | Not yet produced | parity and citation-resolution report | #21B, #22B, #27 | Not passed; docs are not proof. |
| #21B / #22B | Not yet produced | CLI/service and provider integration evidence | #27 | Not passed; docs are not proof. |
| #27 | Not yet produced | answer/citation/abstention verification | #25 | Not passed; docs are not proof. |
| #25 GUI/no-GUI handoff | Not yet produced | immutable issue reference or dated no-GUI decision hash | #26C/#26D | Not passed; docs are not proof. |

Every cross-repository handoff is reciprocal: producer and consumer comments
must each name the other issue, producer commit SHA, schema/API version,
package version, exact compatibility command and passing result, immutable
evidence location, and consumer acceptance. Use this template:

```text
Producer: issue number; commit: exact 40-hex SHA; schema/API: released version; package: name@version
Evidence: immutable path or URL; command: exact command; result: PASS or FAIL
Consumer: issue number; accepted scope/gates: canonical gate names; rollback: named boundary
```

## Interface ownership

- `.memory` #5 owns retrieval, search, hydration, and evidence IDs.
- #21 owns CLI/service ports, transport, and categorized errors.
- #22 owns provider capabilities, status, requests, streaming, and cancellation.
- #27 owns answer, citation, and abstention validation.

No owner may redefine another owner's contract; consumers adapt to the pinned
producer contract through the reciprocal handoff.

## Secondary-gate rule

One packet may satisfy another issue only when every secondary gate is named
before work, verification runs exactly for each named secondary gate, and a
separate evidence comment is posted for each secondary issue. Independent
approval, rollback, or scope means a separate packet.

## Legacy closure routes

| Issue | Maintained default | Retirement default | Closure route |
|---|---|---|---|
| #5 | #27 maintained request boundary | #25 retires unsafe legacy builders | guarded-context regression; #27 does not replace it |
| #7 | after #22B freezes supported clients, create a dedicated packet only if Chroma UI remains supported | close from #25 retirement proof | fresh-path/import regression or verified retirement |
| #8 | #26A early hygiene | #26C final supported-path proof | maintained-entry-point coverage or verified retirement |
| #10 | `.memory` #5/#23 lexical replacement | dedicated legacy-cache packet only if legacy BM25 remains supported | full-corpus regression or verified replacement |
| #11 | carry model provenance through #19/#20 and test the maintained legacy caller | close from retirement proof | mismatch regression or verified retirement |
| #12 | decide at #22B; per-session worker packet only if speech remains supported | retirement | worker-lifecycle regression or verified retirement |
| #13 | #22/#27 maintained request budgeting | #25 retires remaining regex builders | active-path budget regression or verified retirement |
| #16 | 16A via #2A safe environment setup; 16B via #26C after #21/#22 freeze supported entry points | n/a | its named 16A/16B gate |

## Tracked issues

| Issue | Priority and type | Status | Canonical plan | Dependencies and sequence | GitHub |
|---|---|---|---|---|---|
| #2 | P0 security | Open — immediate containment blocker | [Rotate Groq keys](ISSUE-002-rotate-groq-keys.md) | Containment/rotation unlocks #18; early #26B alone rewrites history; #2 closes on its post-rewrite evidence before #19. | [Issue #2](https://github.com/sriharshaguthikonda/easy-local-rag/issues/2) |
| #5 | P1 security | Open — partial implementation | [Prompt-injection defense](ISSUE-005-prompt-injection.md) | Retained safety requirement; its structured validation continues in #27. | [Issue #5](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5) |
| #6 | P0 bug | **Open — fix merged; plan-link closure pending** | [MMR embedding fix](ISSUE-006-mmr-embedding-keyerror.md) | PR #29 merged at `de74659`; preserve #11's embedding contract and close after the main-line plan/evidence comment is available. | [Issue #6](https://github.com/sriharshaguthikonda/easy-local-rag/issues/6) |
| #7 | P1 bug | Open — partial implementation | [Lazy Chroma initialization](ISSUE-007-lazy-chromadb-init.md) | Depends on #11 creation metadata; required only while the Chroma path remains supported. | [Issue #7](https://github.com/sriharshaguthikonda/easy-local-rag/issues/7) |
| #8 | P1 bug | Open — partial implementation | [Config-driven paths](ISSUE-008-config-driven-paths.md) | Coordinates with #16; early #26 owns repository hygiene, not user-path migration. | [Issue #8](https://github.com/sriharshaguthikonda/easy-local-rag/issues/8) |
| #9 | P0 security | Open — partial implementation | [Conversation import validation](ISSUE-009-validate-conversation-import.md) | Independent small closure; imported sessions never restore trusted retrieval state. | [Issue #9](https://github.com/sriharshaguthikonda/easy-local-rag/issues/9) |
| #10 | P1 performance | Open — needs proof | [Full-corpus BM25](ISSUE-010-full-corpus-bm25.md) | Retained by `.memory` #5 and measured by #23; follows the embedding contract in #11. | [Issue #10](https://github.com/sriharshaguthikonda/easy-local-rag/issues/10) |
| #11 | P1 bug | Open — partial implementation | [Embedding-model contract](ISSUE-011-embedding-model-match.md) | Governs #7/#10 and migration provenance in #19/#20; no silent model drift. | [Issue #11](https://github.com/sriharshaguthikonda/easy-local-rag/issues/11) |
| #12 | P1 bug | Open — partial implementation | [Streamlit TTS worker](ISSUE-012-streamlit-tts-worker.md) | Legacy client closure only; future provider/UI behavior is governed by #22 and future GUI work. | [Issue #12](https://github.com/sriharshaguthikonda/easy-local-rag/issues/12) |
| #13 | P1 bug | Open — partial implementation | [Tokenizer budgeting](ISSUE-013-tiktoken-counter.md) | #16 carries the dependency; #27 extends the requirement into structured answer validation. | [Issue #13](https://github.com/sriharshaguthikonda/easy-local-rag/issues/13) |
| #14 | P1 enhancement | **Closed — duplicate/superseded** | [Inline citations](ISSUE-014-inline-citations.md) | Retained as closure evidence; #27 owns citation-resolution validation and abstention. | [Issue #14](https://github.com/sriharshaguthikonda/easy-local-rag/issues/14) |
| #15 | P0 bug | Open — partial implementation | [Atomic vault write](ISSUE-015-atomic-vault-write.md) | Depends on #8; #20A replaces production append behavior with deterministic atomic packages. | [Issue #15](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15) |
| #16 | P1 developer experience | Open — split into 16A/16B | [Setup and dependency docs](ISSUE-016-dx-setup-docs.md) | 16A adds secret-safe contributor setup early; 16B freezes supported entry points/dependencies only after #21/#22. | [Issue #16](https://github.com/sriharshaguthikonda/easy-local-rag/issues/16) |
| #17 | P0 enhancement | Open — canonical parent epic | [PostgreSQL consolidation](ISSUE-017-postgres-consolidation.md) | Governs the full required execution order and the #18-#27 closure gates. | [Issue #17](https://github.com/sriharshaguthikonda/easy-local-rag/issues/17) |
| #18 | P0 security | Open — early preservation gate | [Branch and PR inventory](ISSUE-018-branch-pr-inventory.md) | After #2 containment; before early #26, #19, and any branch/PR deletion. | [Issue #18](https://github.com/sriharshaguthikonda/easy-local-rag/issues/18) |
| #19 | P0 todo | Open — first migration implementation | [Chroma audit and export](ISSUE-019-chroma-audit-export.md) | #19A audit then #19B export follow #18's inventory/proposed disposition, #26A, separately approved #26B evidence, and #2 closure. | [Issue #19](https://github.com/sriharshaguthikonda/easy-local-rag/issues/19) |
| #20 | P1 enhancement | Open — split into A and B | [Deterministic ingestion](ISSUE-020-deterministic-ingestion.md) | #20A follows #19; `.memory` #4 follows #20A; #20B follows `.memory` #4. | [Issue #20](https://github.com/sriharshaguthikonda/easy-local-rag/issues/20) |
| #21 | P1 enhancement | Open — blocked | [Thin CLI client](ISSUE-021-thin-cli-service-client.md) | #21A ports/transport/errors follows `.memory` #5; #21B integration follows #23, before #27. | [Issue #21](https://github.com/sriharshaguthikonda/easy-local-rag/issues/21) |
| #22 | P1 security | Open — provider-policy gate | [Provider modes](ISSUE-022-provider-modes.md) | #22A capabilities/status/requests/streaming/cancellation follows `.memory` #5; #22B integration follows #23, before #27. | [Issue #22](https://github.com/sriharshaguthikonda/easy-local-rag/issues/22) |
| #23 | P0 todo | Open — migration parity gate | [Retrieval parity](ISSUE-023-retrieval-parity.md) | After #21A and #22A; before #21B/#22B. | [Issue #23](https://github.com/sriharshaguthikonda/easy-local-rag/issues/23) |
| #24 | P2 enhancement | Open — specification only | [Thin GUI specification](ISSUE-024-thin-gui-spec.md) | After #21B/#22B and before #27; #25 owns a later GUI/no-GUI handoff after cutover. | [Issue #24](https://github.com/sriharshaguthikonda/easy-local-rag/issues/24) |
| #25 | P0 todo | Open — cutover/retirement gate | [Chroma retirement](ISSUE-025-chroma-retirement.md) | After #27 and all migration/client gates; final state creates the blocked GUI issue or records no-GUI, then unlocks late #26. | [Issue #25](https://github.com/sriharshaguthikonda/easy-local-rag/issues/25) |
| #26 | P1 developer experience | Open — split early/late | [Repository cleanup](ISSUE-026-repo-cleanup.md) | 26A/26B follow #2 containment/#18 and close #2 before #19; 26C/26D consume #25's immutable GUI/no-GUI handoff. | [Issue #26](https://github.com/sriharshaguthikonda/easy-local-rag/issues/26) |
| #27 | P0 security | Open — answer-safety gate | [Grounded answers](ISSUE-027-grounded-answers.md) | Extends #5/#13 and owns structured validation associated with closed duplicate #14; after #24 and #21B/#22B, before #25. | [Issue #27](https://github.com/sriharshaguthikonda/easy-local-rag/issues/27) |

## Gate discipline

- No branch, PR, Chroma state, or rollback artifact is deleted by planning work.
- No PostgreSQL cutover or Chroma retirement occurs without the explicit
  approvals and measurable gates in #25.
- No future GUI implementation starts under #24; open a separate issue after
  #25 succeeds.
- Update this index whenever a tracked issue changes live state or its closure
  evidence changes.
