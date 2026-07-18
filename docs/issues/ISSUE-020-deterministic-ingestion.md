# Issue #20: Deterministic ingestion, chunking, and change detection

**Status:** Open; split into #20A and #20B.
**GitHub:** [#20](https://github.com/sriharshaguthikonda/easy-local-rag/issues/20)
**Parent epic:** #17.
**Labels / priority:** `priority:P1`, `type:enhancement`.

## Dependencies

Requires #19's export contract. **#20A is database-neutral and must complete before `.memory` #4.** `.memory` #4 then establishes the evidence/source store and import contract. **#20B begins only after `.memory` #4** and integrates its stable destination API. `.memory` #5 follows #20B; then #23, #21/#22, #27, #25, future GUI, and late #26 follow in order.

## Implementation slices

### #20A: offline, database-neutral package

1. Discover explicit configured roots/globs; fingerprint canonical source identity and content.
2. Parse text/Markdown, HTML, PDF with page provenance, DOCX, Chroma export, then optional PubMed into one normalized source structure.
3. Chunk deterministically with versioned strategy; validate metadata/text; emit an atomic ingestion package and reconciliation intent without database writes.
4. Define lifecycle intent for no-op, changed, rename-with-same-hash, deletion tombstone, partial parse, interruption, and duplicate-content/different-path cases.

### #20B: destination integration

1. Adapt the package to `.memory` #4's evidence/source import API, never directly to Chroma/PostgreSQL.
2. Import idempotently; atomically supersede/deactivate old source versions; preserve rename lineage and soft-delete semantics.
3. Run embedding as a separate model-keyed stage, then persist reconciliation evidence for #23/#25.

## Affected interfaces, files, and artifacts

- Future package areas: normalized source/chunk model, adapters, deterministic chunker, validation, package writer, lifecycle reconciler, and `.memory` importer adapter.
- Checked-in configuration schema and `.env.example`-style machine-value boundary: roots, globs, size limit, parser/chunker options, embedding profile, destination endpoint, redaction settings; no personal absolute paths in source.
- Package artifact: normalized source/chunk records with source/chunk IDs, hashes, parser/chunker versions, offsets, neighbour identities, warnings, and reconciliation manifest.
- Tests: repeat identity, HTML/section provenance, PDF pages, Unicode/malformed input, lifecycle changes, interrupted recovery, duplicate content, chunker version changes, and adapter no-write guarantee.

## Concrete actions

- Derive IDs from stable source identity/content and versioned parser/chunker inputs. Identical bytes and versions produce identical source/chunk IDs; a strategy change creates a new versioned set.
- Replace append-mode `vault.json` production behavior with atomic package writes; label retired scripts legacy after replacement.
- Enforce embedding model provenance; embeddings are typed stage output, never metadata blobs.
- Keep parsing/chunking offline and database-free; only #20B invokes the shared destination contract.

## Verification

```powershell
python -m pytest tests -q
python -m pytest tests -q -k "ingest or chunk or lifecycle or adapter"
```

Run #20A twice against identical fixtures and compare IDs/hashes/packages. Run #20B twice against the same package through a fake or test `.memory` importer and prove idempotent state plus explicit rename/delete/supersession results.

## Closure gate

#20 closes only when #20A is offline/database-free and deterministic, `.memory` #4 has accepted the package/import boundary, and #20B imports resumably/idempotently with source-resolvable provenance, lifecycle reconciliation, and separate embedding provenance.

## Rollback constraints

No importer may destroy source packages or overwrite an old version in place. Failed imports leave a resumable package and an explicit incomplete state; source deletes are tombstones until #25. Retain old chunks for rollback/audit until the cutover gate accepts retirement.

## Commit boundary

Commit #20A contracts/adapters/package tests separately from #20B destination integration. Do not combine `.memory` schema/API changes or generated packages with either commit.
