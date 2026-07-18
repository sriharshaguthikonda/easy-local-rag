# Canonical issue-plan index

This index is the authoritative execution map for the 24 tracked issues:
[#2](https://github.com/sriharshaguthikonda/easy-local-rag/issues/2) and
[#5](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5) through
[#27](https://github.com/sriharshaguthikonda/easy-local-rag/issues/27).
GitHub remains authoritative for live issue state and discussion; the linked
plan governs implementation scope, gates, verification, rollback, and commit
boundaries. The status column reflects live state checked on 2026-07-18:
#14 is closed, while #6 has a merged GUI-lineage fix and remains open only until this canonical plan lands and receives reciprocal evidence links.

## Required execution order

`#2 containment -> #18 -> early #26 (26A/26B) -> #2 final closure -> #19 -> #20A -> .memory #4 -> #20B -> .memory #5 -> #23 -> #21/#22 -> #27 -> #25 -> future GUI decision/implementation -> late #26 (26C/26D)`

#24 is a specification-only issue. It records the future GUI contract and
salvage decisions; it does not authorize GUI implementation or alter the
required order above. #17 is the parent epic. Legacy issues #5-#16 retain their
own closure gates but must not bypass the consolidation sequence.

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
| #19 | P0 todo | Open — first migration implementation | [Chroma audit and export](ISSUE-019-chroma-audit-export.md) | After #18, the early 26A/26B handoff, and #2 final closure; blocks #20A and all destination work. | [Issue #19](https://github.com/sriharshaguthikonda/easy-local-rag/issues/19) |
| #20 | P1 enhancement | Open — split into A and B | [Deterministic ingestion](ISSUE-020-deterministic-ingestion.md) | #20A follows #19; `.memory` #4 follows #20A; #20B follows `.memory` #4. | [Issue #20](https://github.com/sriharshaguthikonda/easy-local-rag/issues/20) |
| #21 | P1 enhancement | Open — blocked | [Thin CLI client](ISSUE-021-thin-cli-service-client.md) | After #20B, `.memory` #5, and #23; runs alongside #22 before #27. | [Issue #21](https://github.com/sriharshaguthikonda/easy-local-rag/issues/21) |
| #22 | P1 security | Open — provider-policy gate | [Provider modes](ISSUE-022-provider-modes.md) | After #20B, `.memory` #5, #23, and the thin service boundary; before #27. | [Issue #22](https://github.com/sriharshaguthikonda/easy-local-rag/issues/22) |
| #23 | P0 todo | Open — migration parity gate | [Retrieval parity](ISSUE-023-retrieval-parity.md) | After #19/#20A/`.memory` #4/#20B/`.memory` #5; before #21/#22. | [Issue #23](https://github.com/sriharshaguthikonda/easy-local-rag/issues/23) |
| #24 | P2 enhancement | Open — specification only | [Thin GUI specification](ISSUE-024-thin-gui-spec.md) | Closes on the spec/PR #1 disposition before #25; #25 owns opening a later implementation issue after #22 and cutover pass. | [Issue #24](https://github.com/sriharshaguthikonda/easy-local-rag/issues/24) |
| #25 | P0 todo | Open — cutover/retirement gate | [Chroma retirement](ISSUE-025-chroma-retirement.md) | After #27 and all migration/client gates; future GUI follows successful cutover. | [Issue #25](https://github.com/sriharshaguthikonda/easy-local-rag/issues/25) |
| #26 | P1 developer experience | Open — split early/late | [Repository cleanup](ISSUE-026-repo-cleanup.md) | 26A/26B follow #2 containment/#18 and close #2 before #19; 26C/26D follow #25 and the future GUI decision. | [Issue #26](https://github.com/sriharshaguthikonda/easy-local-rag/issues/26) |
| #27 | P0 security | Open — answer-safety gate | [Grounded answers](ISSUE-027-grounded-answers.md) | Extends #5/#13 and owns structured validation associated with closed duplicate #14; after #21/#22, before #25. | [Issue #27](https://github.com/sriharshaguthikonda/easy-local-rag/issues/27) |

## Gate discipline

- No branch, PR, Chroma state, or rollback artifact is deleted by planning work.
- No PostgreSQL cutover or Chroma retirement occurs without the explicit
  approvals and measurable gates in #25.
- No future GUI implementation starts under #24; open a separate issue after
  #25 succeeds.
- Update this index whenever a tracked issue changes live state or its closure
  evidence changes.
