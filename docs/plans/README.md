# Historical PostgreSQL migration notes

This directory is supporting historical context. The authoritative issue/plan/status
index, sequencing, and closure gates are in the
[roadmap ledger](../issues/README.md) and its per-issue plans; this directory
is not a second authority.
Where these notes differ, `docs/issues/` governs.

## Target boundary

`easy-local-rag` will own document adapters, deterministic parsing and chunking, Chroma audit/export tools, retrieval evaluation fixtures, and a thin command-line client.

The `.memory` repository will own the PostgreSQL/pgvector evidence store and shared retrieval boundary. Durable memories and imported evidence remain separate domains. Raw document chunks must not be inserted into durable-memory tables or injected into every prompt.

## Execution order

### 0. Contain and freeze

1. Pass [#2](https://github.com/sriharshaguthikonda/easy-local-rag/issues/2)'s credential containment gate. Never publish secret values in reports or issue comments.
2. Treat every non-default branch as read-only salvage material until [#18](https://github.com/sriharshaguthikonda/easy-local-rag/issues/18) records its SHA, risks, useful ideas, and proposed disposition; #26A follows, then separately approved #26B solely owns any history rewrite and supplies #2 final-closure evidence.
3. Do not merge PR #1 or either GUI branch wholesale.

### 1. Preserve the current evidence

1. Finish the public and local branch inventory in [#18](https://github.com/sriharshaguthikonda/easy-local-rag/issues/18), then complete #26A.
2. Perform only the separately approved #26B history rewrite.
3. Close #2 from the canonical clean or documented, explicitly accepted
   immutable-ref residual outcome.
4. Build #19A's read-only Chroma audit, then #19B's deterministic export tool in [#19](https://github.com/sriharshaguthikonda/easy-local-rag/issues/19).
5. Freeze a named Chroma snapshot and record checksums before any migration write.

### 2. Define data contracts

1. Implement #20A's database-neutral discovery, fingerprinting, parsing, chunking, metadata, and package contract in [#20](https://github.com/sriharshaguthikonda/easy-local-rag/issues/20).
2. Implement the PostgreSQL evidence/source boundary in `.memory` issue #4 against that frozen contract.
3. Implement #20B's destination integration and reconciliation.
4. Require stable source IDs, chunk IDs, content hashes, parser/chunker versions, and idempotent imports.

### 3. Prove retrieval before changing clients

1. Implement independent lexical, vector, and metadata candidate retrieval in `.memory` issue #5.
2. Define #21A CLI/service ports/transport/errors and #22A provider capabilities/status/requests/streaming/cancellation.
3. Build Chroma-to-PostgreSQL parity tests in [#23](https://github.com/sriharshaguthikonda/easy-local-rag/issues/23).
4. Do not call a vector-first rerank "hybrid retrieval"; lexical and vector searches must form independent candidate sets before fusion.

### 4. Move the useful interface

1. Integrate #21B's thin client and #22B's explicit offline-search, local-chat, and cloud-chat provider modes.
2. Close [#24](https://github.com/sriharshaguthikonda/easy-local-rag/issues/24) as the GUI specification only.
3. Add structured evidence-grounding, citation validation, and abstention in [#27](https://github.com/sriharshaguthikonda/easy-local-rag/issues/27).

### 5. Cut over, then clean up

1. Use the dual-run and rollback gates in [#25](https://github.com/sriharshaguthikonda/easy-local-rag/issues/25).
2. Keep Chroma read-only until counts, hashes, provenance, retrieval quality, and rollback have passed.
3. After cutover, record the immutable GUI/no-GUI handoff: open the separate future GUI implementation issue from [#24](https://github.com/sriharshaguthikonda/easy-local-rag/issues/24)'s specification, or record an explicit no-GUI decision.
4. Perform late #26C, then #26D dependency, legacy-layout, and Chroma-removal cleanup.

## Stop conditions

Stop the migration or merge when any of these occurs:

- a credential is found and has not been revoked or rotated;
- an operation would modify the source Chroma database;
- export/import counts or content hashes cannot be reconciled;
- embedding model or chunking provenance is unknown and treated as known;
- lexical retrieval is derived only from vector candidates;
- evidence IDs or citations cannot be resolved to the exact context sent to the model;
- a client silently falls back from local to cloud;
- a branch, database, or rollback copy would be deleted without an immutable recorded reference.

## Supporting records

- [Public branch and PR inventory](branch-inventory.md)
- [Issue #2–#27 disposition map](issue-disposition.md)
- [Superseded Qwen3 embedding proposal](../proposals/qwen3-embedding-migration.md)
- [Canonical issue-plan index](../issues/README.md) and [#2](../issues/ISSUE-002-rotate-groq-keys.md), [#5](../issues/ISSUE-005-prompt-injection.md), [#6](../issues/ISSUE-006-mmr-embedding-keyerror.md), [#7](../issues/ISSUE-007-lazy-chromadb-init.md), [#8](../issues/ISSUE-008-config-driven-paths.md), [#9](../issues/ISSUE-009-validate-conversation-import.md), [#10](../issues/ISSUE-010-full-corpus-bm25.md), [#11](../issues/ISSUE-011-embedding-model-match.md), [#12](../issues/ISSUE-012-streamlit-tts-worker.md), [#13](../issues/ISSUE-013-tiktoken-counter.md), [#14](../issues/ISSUE-014-inline-citations.md), [#15](../issues/ISSUE-015-atomic-vault-write.md), [#16](../issues/ISSUE-016-dx-setup-docs.md), [#17](../issues/ISSUE-017-postgres-consolidation.md), [#18](../issues/ISSUE-018-branch-pr-inventory.md), [#19](../issues/ISSUE-019-chroma-audit-export.md), [#20](../issues/ISSUE-020-deterministic-ingestion.md), [#21](../issues/ISSUE-021-thin-cli-service-client.md), [#22](../issues/ISSUE-022-provider-modes.md), [#23](../issues/ISSUE-023-retrieval-parity.md), [#24](../issues/ISSUE-024-thin-gui-spec.md), [#25](../issues/ISSUE-025-chroma-retirement.md), [#26](../issues/ISSUE-026-repo-cleanup.md), and [#27](../issues/ISSUE-027-grounded-answers.md).

## Current authority

Use [the canonical issue-plan index](../issues/README.md) for immediate work.
Issue #14 is already closed as a duplicate; its historical evidence is retained
under `docs/issues/`, while #27 owns structured answer validation.
