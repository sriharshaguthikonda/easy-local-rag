# Historical issue #2–#27 disposition snapshot

This snapshot is supporting branch-salvage context, not a current status or
sequencing authority. It is superseded by the
[canonical issue-plan index](../issues/README.md) and the per-issue plans in
`docs/issues/`. The aligned dispositions below preserve each issue's own
closure gate unless GitHub has explicitly closed it as a duplicate.

| Issue | Disposition | Canonical destination |
|---|---|---|
| #2 exposed credentials | **Keep — blocking** | Rotation/revocation, all-ref scan, history remediation; also gates #26. Never publish values. |
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
| #18 branch inventory/archive | **Keep — early** | Complete public and local inventories, sanitised archives, and PR #1 disposition. |
| #19 read-only Chroma audit/export | **Keep — first implementation** | Must precede schema migration or Chroma cleanup. |
| #20 deterministic ingestion contracts | **Keep** | Build #20A before `.memory` #4 and #20B after it. |
| #21 thin CLI client | **Keep** | Start after retrieval contracts exist; the old CLI is the first supported client. |
| #22 provider abstraction/modes | **Keep** | Provider core can proceed beside #21 after the shared service contract; integration follows #21. |
| #23 retrieval parity/regression | **Keep — migration gate** | Required before PostgreSQL becomes primary. |
| #24 future thin GUI | **Keep, defer** | No wholesale reuse; begin only after #21, #23 and #27 are stable. |
| #25 dual-run/rollback/retirement | **Keep — cutover gate** | Chroma remains read-only rollback until all gates pass. |
| #26 history/layout/generated artefact cleanup | **Keep, split** | Early hygiene follows #2/#18; late layout/Chroma cleanup follows #25 and the future GUI decision. |
| #27 grounded answers/citation validation | **Keep** | Replaces #5 and #14 as the answer-safety contract. |

## Recommended issue sequencing

1. #2 credential containment and rotation.
2. #18 public/local inventory and PR disposition.
3. Early #26 generated/private-artifact hygiene.
4. #19 read-only audit/export.
5. #20A database-neutral ingestion package.
6. `.memory` #4 evidence schema/import boundary.
7. #20B destination integration and reconciliation.
8. `.memory` #5 retrieval APIs.
9. #23 parity/regression suite.
10. #21/#22 thin-client and provider work after the shared contract.
11. #27 grounded answers.
12. #25 dual-run/cutover gate.
13. Future GUI implementation issue, if still wanted.
14. Late #26 layout/Chroma cleanup.

## Closure policy for legacy issues

Close #5–#16 only through each issue's canonical closure gate, except #14,
which GitHub already closed as a duplicate. A closure requires:

- its fix is part of a supported path and has a regression test;
- the affected path is archived and marked unsupported, with the replacement linked.

A commit on a stale or deleted experimental ref is not enough to prove the maintained path is fixed.
