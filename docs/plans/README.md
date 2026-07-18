# Historical PostgreSQL migration notes

This directory is supporting historical context. The authoritative issue status,
sequencing, and closure gates are in the
[canonical issue-plan index](../issues/README.md) and its per-issue plans.
Where these notes differ, `docs/issues/` governs.

## Target boundary

`easy-local-rag` will own document adapters, deterministic parsing and chunking, Chroma audit/export tools, retrieval evaluation fixtures, and a thin command-line client.

The `.memory` repository will own the PostgreSQL/pgvector evidence store and shared retrieval boundary. Durable memories and imported evidence remain separate domains. Raw document chunks must not be inserted into durable-memory tables or injected into every prompt.

## Execution order

### 0. Contain and freeze

1. Complete credential rotation and history remediation in [#2](https://github.com/sriharshaguthikonda/easy-local-rag/issues/2). Never publish secret values in reports or issue comments.
2. Treat every non-default branch as read-only salvage material until [#18](https://github.com/sriharshaguthikonda/easy-local-rag/issues/18) records its SHA, risks, useful ideas, and disposition.
3. Do not merge PR #1 or either GUI branch wholesale.

### 1. Preserve the current evidence

1. Finish the public and local branch inventory in [#18](https://github.com/sriharshaguthikonda/easy-local-rag/issues/18).
2. Perform only the early generated/private-artifact hygiene slice of [#26](https://github.com/sriharshaguthikonda/easy-local-rag/issues/26).
3. Build the read-only Chroma audit and deterministic export tool in [#19](https://github.com/sriharshaguthikonda/easy-local-rag/issues/19).
4. Freeze a named Chroma snapshot and record checksums before any migration write.

### 2. Define data contracts

1. Implement #20A's database-neutral discovery, fingerprinting, parsing, chunking, metadata, and package contract in [#20](https://github.com/sriharshaguthikonda/easy-local-rag/issues/20).
2. Implement the PostgreSQL evidence/source boundary in `.memory` issue #4 against that frozen contract.
3. Implement #20B's destination integration and reconciliation.
4. Require stable source IDs, chunk IDs, content hashes, parser/chunker versions, and idempotent imports.

### 3. Prove retrieval before changing clients

1. Implement independent lexical, vector, and metadata candidate retrieval in `.memory` issue #5.
2. Build Chroma-to-PostgreSQL parity tests in [#23](https://github.com/sriharshaguthikonda/easy-local-rag/issues/23).
3. Do not call a vector-first rerank "hybrid retrieval"; lexical and vector searches must form independent candidate sets before fusion.

### 4. Move the useful interface

1. After the shared service contract is stable, refactor the old command-line workflow into a thin client in [#21](https://github.com/sriharshaguthikonda/easy-local-rag/issues/21) while provider-core work proceeds in parallel in [#22](https://github.com/sriharshaguthikonda/easy-local-rag/issues/22).
2. Integrate explicit offline-search, local-chat, and cloud-chat provider modes after #21 exposes the shared client surface.
3. Add structured evidence-grounding, citation validation, and abstention in [#27](https://github.com/sriharshaguthikonda/easy-local-rag/issues/27).

### 5. Cut over, then clean up

1. Use the dual-run and rollback gates in [#25](https://github.com/sriharshaguthikonda/easy-local-rag/issues/25).
2. Keep Chroma read-only until counts, hashes, provenance, retrieval quality, and rollback have passed.
3. Rebuild a thin GUI only after the shared contracts are stable, as described in [#24](https://github.com/sriharshaguthikonda/easy-local-rag/issues/24).
4. Perform only the late legacy-layout and Chroma-removal slice of [#26](https://github.com/sriharshaguthikonda/easy-local-rag/issues/26).

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

## Current authority

Use [the canonical issue-plan index](../issues/README.md) for immediate work.
Issue #14 is already closed as a duplicate; its historical evidence is retained
under `docs/issues/`, while #27 owns structured answer validation.
