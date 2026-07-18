# Issue #2–#27 disposition map

This map prevents old experimental fixes from driving the PostgreSQL migration. `Keep` means the issue remains a direct deliverable. `Absorb` means its acceptance criteria move into a newer issue. `Supersede` means the old implementation-specific issue should close after the linked replacement is recorded.

| Issue | Disposition | Canonical destination |
|---|---|---|
| #2 exposed credentials | **Keep — blocking** | Rotation/revocation, all-ref scan, history remediation; also gates #26. Never publish values. |
| #3 subprocess injection | Closed on experimental history; **retain as regression requirement** | #21 client boundary and #24 GUI rules. Old commit is not on `main`. |
| #4 arbitrary file open | Closed on experimental history; **retain as regression requirement** | #21/#24. Any future opener needs allowlisted roots/types and no shell fallback. |
| #5 prompt injection | **Absorb** | #27 context fencing, untrusted-evidence rules, citation validation and abstention. |
| #6 missing metadata embedding/MMR crash | **Legacy implementation defect** | #23 must test embeddings as typed retrieval data, not Chroma metadata. Close when the legacy path is frozen. |
| #7 eager Chroma collection initialisation | **Supersede** | #24 thin client; clients must not initialise storage during import. |
| #8 hard-coded Windows paths | **Absorb** | #20 explicit source configuration and #26 repository cleanup. |
| #9 Streamlit state import hijack | **Supersede with security requirement** | #24. Reintroduce conversation import only through a versioned allowlisted schema. |
| #10 BM25 over vector top-50 | **Supersede** | `.memory` #5 independent lexical/vector retrieval plus #23 parity tests. |
| #11 embedding-model mismatch | **Absorb** | #20 embedding provenance and #23 mismatch tests. |
| #12 Streamlit TTS worker deadlock | **Supersede** | #22 provider adapters and #24 thin UI lifecycle. |
| #13 regex token counter/provider overflow | **Absorb** | #22 provider-specific limits and #27 exact context/truncation reporting. |
| #14 prose-only inline citations | **Close as superseded** | #27 structured claim/citation contract, validation and abstention. |
| #15 corrupt append-only `vault.json` | **Absorb** | #20 deterministic packages and atomic/idempotent writes; retire the vault as a production store. |
| #16 `.env.example`, AGENTS and dependency lock | **Split, keep open for now** | Add safe contributor/env guidance early; finish the supported dependency lock after #21/#22 define the maintained package surface. Broader cleanup belongs to #26. |
| #17 PostgreSQL/pgvector consolidation | **Epic — canonical** | Sequenced by `docs/plans/README.md`. |
| #18 branch inventory/archive | **Keep — early** | Complete public and local inventories, sanitised archives, and PR #1 disposition. |
| #19 read-only Chroma audit/export | **Keep — first implementation** | Must precede schema migration or Chroma cleanup. |
| #20 deterministic ingestion contracts | **Keep** | Build after the audit format and PostgreSQL evidence boundary are known. |
| #21 thin CLI client | **Keep** | Start after retrieval contracts exist; the old CLI is the first supported client. |
| #22 provider abstraction/modes | **Keep** | Follows the thin service boundary; no silent cloud fallback. |
| #23 retrieval parity/regression | **Keep — migration gate** | Required before PostgreSQL becomes primary. |
| #24 future thin GUI | **Keep, defer** | No wholesale reuse; begin only after #21, #23 and #27 are stable. |
| #25 dual-run/rollback/retirement | **Keep — cutover gate** | Chroma remains read-only rollback until all gates pass. |
| #26 history/layout/generated artefact cleanup | **Keep, last** | Blocked by #2, #18, #19, #23 and #25. |
| #27 grounded answers/citation validation | **Keep** | Replaces #5 and #14 as the answer-safety contract. |

## Recommended issue sequencing

1. #2 containment work in parallel with the **non-destructive** parts of #18.
2. #18 public/local inventory, sanitised archive decisions and PR #1 closure.
3. #19 read-only audit/export.
4. `.memory` #4 evidence schema/import boundary.
5. #20 deterministic ingestion and reconciliation.
6. `.memory` #5 retrieval APIs.
7. #23 parity/regression suite.
8. #21 thin CLI.
9. #22 provider modes and #27 grounded answers.
10. #25 dual-run/cutover.
11. #24 optional GUI.
12. #26 cleanup and retirement.

## Closure policy for legacy issues

Do not patch the experimental GUI merely to close #5–#15. Close an old issue only when one of these is true:

- its fix is part of a supported path and has a regression test;
- its requirement is explicitly accepted by a newer canonical issue;
- the affected path is archived and marked unsupported, with the replacement linked.

A commit on a stale or deleted experimental ref is not enough to prove the maintained path is fixed.