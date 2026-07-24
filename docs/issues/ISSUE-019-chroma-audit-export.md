# Issue #19: Read-only Chroma audit and deterministic export

[Roadmap ledger](../plans/README.md)

**Status:** Open; first migration implementation.
**GitHub:** [#19](https://github.com/sriharshaguthikonda/easy-local-rag/issues/19)
**Parent epic:** #17.
**Labels / priority:** `priority:P0`, `type:todo`.

## Dependencies

Requires #18's immutable inventory and proposed disposition plus #26B evidence,
and #2 final closure from #26B's clean or explicitly accepted
immutable-residual terminal evidence. Blocks #20A, `.memory` #4, #20B, #23,
#25, and late #26. Chroma remains untouched and available as rollback data.

## Implementation slices

1. **Read-only audit:** discover all collections dynamically, including historical names; inspect counts, IDs, metadata schema/types, source/chunk identity, duplicates, timestamps, malformed records, and embeddings by requesting typed embeddings rather than metadata.
2. **Snapshot consistency:** open against read-only storage or a filesystem snapshot; detect count changes and fail without a snapshot.
3. **Versioned export:** write one JSONL or Parquet record file per collection, optional separately typed embeddings, `manifest.json`, checksums, warnings/skips, exporter/version/commit, source fingerprint, and host-independent schema version.
4. **Reproducibility:** canonical record ordering and normalized metadata make repeated unchanged-snapshot exports content-identical except explicitly excluded run metadata.

## Affected interfaces, files, and artifacts

- New modules: `easy_local_rag.chroma_audit` and `easy_local_rag.chroma_export` (or the established package-equivalent module paths).
- CLI: `python -m easy_local_rag.chroma_audit` and `python -m easy_local_rag.chroma_export`; options `--path`, `--output`, `--snapshot`, `--dry-run`, `--collection`, `--redact-paths`, `--include-embeddings`, machine-readable exits.
- Export artifact example: `migration-export/manifest.json`, `migration-export/html_chunks.jsonl` (or `.parquet`), optional typed embeddings file, checksums, warnings/skipped-record report.
- Tests: multi-collection, duplicate ID/text, missing/mixed metadata, empty collection, unknown embedding provenance, source mutation, and deterministic repeat export fixtures.

## Concrete actions

- Store raw document, validated metadata JSON, original IDs/collection, source reference, order/timestamps, text/source hashes, and embedding provenance only when trustworthy.
- Redact document bodies, secrets, and private absolute paths by default from diagnostics and samples.
- Permit stored-embedding reuse only with proven model/version, dimension, normalization, metric, exact text hash, and preprocessing identity; otherwise export text for re-embedding.
- Forbid `delete`, `update`, `upsert`, or Chroma metadata rewriting in audit/export paths.

## Verification

```powershell
$chromaSnapshot = 'C:\rag-migration\chroma-snapshot'
$auditOutput = 'C:\rag-migration\audit'
$firstExport = 'C:\rag-migration\export-1'
$secondExport = 'C:\rag-migration\export-2'
python -m pytest tests -q
python -m easy_local_rag.chroma_audit --path $chromaSnapshot --output $auditOutput --dry-run
python -m easy_local_rag.chroma_export --path $chromaSnapshot --output $firstExport --snapshot
python -m easy_local_rag.chroma_export --path $chromaSnapshot --output $secondExport --snapshot
```

Compare canonical record checksums and reconcile each collection's source count/record count with the manifest. Hash the source snapshot before and after to prove unchanged bytes.

## Closure gate

Every discovered collection reconciles exactly with the export; every malformed/skipped record is explicit; unchanged repeated exports have identical canonical checksums; source Chroma bytes are unchanged; and the export contains enough provenance/hashes for idempotent `.memory` import and #25 rollback audit.

## Rollback constraints

The tool must be observational only. On mutation detection or export failure, retain the source unchanged, mark the manifest incomplete, and rerun from a verified snapshot. Never use export output to overwrite the original Chroma store.

## Commit boundary

One commit for read-only audit/export implementation and its focused fixtures; a separate commit for any generated sample manifest only if it is sanitized, deterministic, and intentionally versioned.
