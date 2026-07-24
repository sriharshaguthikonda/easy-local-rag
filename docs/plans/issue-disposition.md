# Historical issue #2–#27 disposition snapshot

This snapshot is supporting branch-salvage context, not a current status or
sequencing authority. It is superseded by the
[canonical issue-plan index](../issues/README.md) and the per-issue plans in
`docs/issues/`. The aligned dispositions below preserve each issue's own
closure gate unless GitHub has explicitly closed it as a duplicate.

| Issue | Disposition | Canonical destination |
|---|---|---|
| #2 exposed credentials | **Keep — blocking** | #2 owns containment/rotation and closes only after #26B supplies either clean all-ref evidence or the documented explicitly accepted immutable-ref residual outcome; #26B alone executes any approved rewrite. Never publish values. |
| #3 subprocess injection | Closed on experimental history; **retain as regression requirement** | #21 client boundary and #24 GUI rules. Old commit is not on `main`. |
| #4 arbitrary file open | Closed on experimental history; **retain as regression requirement** | #21/#24. Any future opener needs allowlisted roots/types and no shell fallback. |
| #5 prompt injection | **Keep open** | Complete its own guarded-context closure gate; #27 extends it with citation validation and abstention. |
| #6 missing metadata embedding/MMR crash | **Keep until linked closure** | GUI-lineage fix `de74659` is merged; close after the main-line plan and evidence comment land. |
| #7 eager Chroma collection initialisation | **Keep open** | Complete its fresh-path/import closure gate; #24 carries the future thin-client rule. |
| #8 hard-coded Windows paths | **Keep open** | Complete retained-entry-point configuration coverage; #20/#26 consume the result. |
| #9 Streamlit state import hijack | **Keep open** | Complete its allowlisted-import regression gate; #24 inherits the contract. |
| #10 BM25 over vector top-50 | **Keep open** | Complete its own full-corpus retrieval gate; `.memory` #5/#23 later preserve parity. |
| #11 embedding-model mismatch | **Keep open** | Complete ingest/query enforcement; #20/#23 reuse the provenance contract. |
| #12 Streamlit TTS worker deadlock | **Keep open** | Complete its worker-lifecycle gate; #22/#24 later consume the boundary. |
| #13 regex token counter/provider overflow | **Keep open** | Complete active-path token-budget coverage; #22/#27 reuse it. |
| #14 prose-only inline citations | **Closed — duplicate** | Historical display evidence only; #27 owns structured validation and abstention. |
| #15 corrupt append-only `vault.json` | **Keep open** | Complete atomic/idempotent failure regressions; #20 later consumes the contract. |
| #16 `.env.example`, AGENTS and dependency lock | **Split, keep open for now** | Add safe contributor/env guidance early; finish the supported dependency lock after #21/#22 define the maintained package surface. Broader cleanup belongs to #26. |
| #17 PostgreSQL/pgvector consolidation | **Epic — canonical** | Sequenced by `docs/issues/README.md`. |
| #18 branch inventory/archive | **Keep — early inventory** | Record immutable public/local ref inventory, pre-rewrite scan evidence, proposed archive dispositions, and PR #1 disposition; #26B alone sanitizes/replaces refs and creates approved archives. |
| #19 read-only Chroma audit/export | **Keep — first implementation** | Must precede schema migration or Chroma cleanup. |
| #20 deterministic ingestion contracts | **Keep** | Build #20A before `.memory` #4 and #20B after it. |
| #21 thin CLI client | **Keep** | Start after retrieval contracts exist; the old CLI is the first supported client. |
| #22 provider abstraction/modes | **Keep** | Provider core can proceed beside #21 after the shared service contract; integration follows #21. |
| #23 retrieval parity/regression | **Keep — migration gate** | Required before PostgreSQL becomes primary. |
| #24 future thin GUI | **Keep, specification only** | Close on the approved specification and PR #1 disposition. Any implementation is a separate post-#25 issue requiring stable #21/#22/#23/#27 contracts. |
| #25 dual-run/rollback/retirement | **Keep — cutover gate** | Chroma remains read-only rollback until all gates pass. |
| #26 history/layout/generated artefact cleanup | **Keep, split** | 26A/26B follow #2 containment/#18 and supply #2 closure evidence before #19; 26C/26D follow #25 and the future GUI decision. |
| #27 grounded answers/citation validation | **Keep** | Extends #5's guarded-context requirement and owns the structured validation/abstention scope associated with closed duplicate #14. |

## Recommended issue sequencing

1. #2 credential containment and rotation (issue remains open).
2. #18 public/local inventory and PR disposition.
3. #26A generated/private-artifact hygiene, then separately approved #26B's sole history rewrite.
4. #2 final closure from post-rewrite all-ref evidence.
5. #19 read-only audit/export.
6. #20A database-neutral ingestion package.
7. `.memory` #4 evidence schema/import boundary.
8. #20B destination integration and reconciliation.
9. `.memory` #5 retrieval APIs.
10. #21A CLI/service ports and #22A provider capabilities.
11. #23 parity/regression suite.
12. #21B CLI/service integration and #22B provider integration.
13. #24 specification only.
14. #27 grounded answers.
15. #25 dual-run/cutover gate and GUI/no-GUI handoff.
16. Late #26C then #26D cleanup.

## Closure policy for legacy issues

Close #5–#16 only through one explicitly selected canonical mode, except #14,
which GitHub already closed as a duplicate:

- **Maintained-path fix:** the supported path passes the issue's regression and
  verification gate.
- **Verified retirement:** the affected entry point is removed from the
  supported surface, blocked from normal invocation, linked to its supported
  replacement, and covered by a regression proving it cannot silently return.

The modes are mutually exclusive for a closure record; evidence must name the
chosen mode.

A commit on a stale or deleted experimental ref is not enough to prove the maintained path is fixed.
