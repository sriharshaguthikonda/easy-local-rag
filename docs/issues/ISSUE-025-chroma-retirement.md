# Issue #25 Plan: Chroma dual-run, rollback and retirement

[Roadmap ledger](../plans/README.md)

Status: **Open — cutover and retirement gate**

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/25

Parent epic: [#17](https://github.com/sriharshaguthikonda/easy-local-rag/issues/17)

Labels: `priority:P0`, `type:todo`

## Goal

Move supported use to PostgreSQL only after deterministic export/import,
reconciliation, retrieval parity, client migration and rollback have been
demonstrated. Chroma remains an immutable rollback source until every gate and
the hold period pass.

## Dependencies and sequence

Cutover cannot start until:

- #19 records the frozen Chroma snapshot, collections and export checksums;
- #20B completes destination import and reconciliation through `.memory` #4;
- `.memory` #5 exposes stable search/hydration;
- #23 passes its retrieval and citation-resolution gate;
- #21B/#22B provide a supported PostgreSQL client with explicit provider modes;
- #27 validates generated answers and preserves search-only operation;
- #24 supplies the specification proving no supported GUI directly owns Chroma.

#25 executes after #27. A future GUI follows successful cutover. Late #26
removes legacy dependencies and reorganizes the repository only after #25.

## Migration state machine

The machine-readable state records these ordered transitions:

1. `frozen`
2. `exported`
3. `imported`
4. `reconciled`
5. `validated`
6. `dual_run`
7. `postgres_primary`
8. `chroma_retired`
9. `archive_approved`
10. `post_cutover_handoff`

Every transition records timestamp, actor, source/export hashes, importer and
exporter commits, migration batch ID, reports, approval and previous state.
Skipping a state is invalid.

## Implementation slices

### 25A — State schema and runbook

- Add a checked-in state schema and human-readable runbook.
- Validate required evidence before each transition.
- Keep the live state file local/generated and free of source text, secrets and
  personal path prefixes.

### 25B — Read-only dual-run

- Freeze Chroma ingestion when PostgreSQL import begins.
- Route normal supported use to PostgreSQL while optionally sampling queries
  against the frozen Chroma backend.
- Compare stable chunk/source IDs, text hashes, top-K overlap and latency.
- Log identifiers and aggregate metrics only, never private document bodies.

### 25C — PostgreSQL-primary hold

- Make PostgreSQL the configured primary only after explicit user approval.
- Retain Chroma snapshot/export and a tested backend-selection rollback.
- For every accepted post-freeze ingestion, retain its deterministic #20 package
  and append it to a checksummed rollback manifest before acknowledging the
  ingest. A rollback replays that manifest into a separate Chroma delta
  collection and searches the immutable frozen snapshot plus the delta; it
  never writes to the frozen snapshot.
- Run for 14 consecutive days without an unexplained P0/P1 parity, citation,
  ingestion or data-loss regression.

### 25D — Runtime retirement and archive approval

- Prove the supported CLI starts and searches with Chroma uninstalled.
- Move Chroma tooling to an optional legacy migration surface.
- Require a second explicit user approval before archive/deletion eligibility.
- Actual local data deletion is a separate manual action, not part of #25.

### 25E — Post-cutover GUI decision handoff

- After cutover and hold gates pass, either create the separate GUI
  implementation issue from #24's handoff text or record an explicit no-GUI
  decision.
- A created issue is blocked by completed #22 and #25 and links the consumed
  #21/#23/#24/#27 contracts; #25 does not implement GUI code.
- Record the immutable issue URL/number or dated no-GUI decision hash in
  migration state, then transition to `post_cutover_handoff` to unlock
  #26C/#26D.

## Affected interfaces, files and artifacts

Planned files:

- `migration_state.py` — transition validation and evidence checks.
- `docs/migration/chroma-retirement-runbook.md` — operator procedure.
- `tests/test_migration_state.py` — state, evidence and unsafe-path tests.
- `docs/issues/ISSUE-025-chroma-retirement.md` — authoritative plan.

Required retained artifacts:

- frozen Chroma snapshot/backup and fingerprint;
- deterministic #19 export and checksums;
- import-batch and reconciliation reports;
- #23 retrieval report;
- rollback demonstration record;
- list of post-freeze sources that exist only in PostgreSQL;
- checksummed post-freeze package manifest and rebuilt rollback delta;
- immutable future-GUI issue link or explicit no-GUI decision record;
- machine-readable migration state.

## Concrete actions

1. Define and test the state schema and transition preconditions.
2. Verify the exact frozen snapshot and export hashes.
3. Attach import counts, quarantine counts and hash reconciliation.
4. Attach the passing #23 report.
5. Exercise dual-run with Chroma opened read-only.
6. Demonstrate rollback from PostgreSQL-primary configuration to the frozen
   Chroma-plus-delta backend, including a source ingested only after freeze.
7. Record approval, begin the 14-day PostgreSQL-primary hold and monitor.
8. Run the supported client with Chroma uninstalled.
9. Record separate archive approval; leave deletion manual.
10. Create the correctly blocked post-cutover GUI issue or record the explicit
    no-GUI decision, persist its immutable reference, and unlock #26C/#26D.

## Verification

```powershell
python -m pytest tests/test_migration_state.py -q
python -m pytest tests -q
python -m py_compile migration_state.py
python migration_state.py verify --state $env:EASY_RAG_MIGRATION_STATE
python retrieval_evaluation.py compare --fail-on-critical-regression
python localrag.py status
python localrag.py search "migration smoke" --json
python -c "import importlib.util; assert importlib.util.find_spec('chromadb') is None"
git diff --check
```

The Chroma-uninstalled smoke runs in a clean environment containing only the
supported PostgreSQL client dependencies.

## Measurable closure gate

- All ten states are recorded in order with immutable evidence references.
- Source/chunk counts and hashes reconcile for every non-quarantined record.
- #23 passes with 100% citation resolution and no unexplained critical
  regression.
- Rollback is executed successfully, not merely documented.
- Rollback has an RPO of zero acknowledged ingestion packages: every accepted
  post-freeze package is present in the checksummed manifest and replayed into
  the delta. A regression fixture proves a post-freeze-only source is
  discoverable with its citation/source location while PostgreSQL is disabled.
- PostgreSQL remains primary for 14 consecutive days without an unexplained
  P0/P1 migration regression.
- The supported CLI searches successfully with Chroma uninstalled.
- `post_cutover_handoff` records either a correctly blocked GUI implementation
  issue URL/number or a dated no-GUI decision hash; this reference is the only
  GUI-decision prerequisite consumed by #26C/#26D.
- Chroma snapshot/export checksums still verify after the hold.
- Two explicit approvals are recorded: one for PostgreSQL-primary cutover and
  one for archive eligibility.

## Rollback and safety constraints

- **No cutover without explicit user approval.**
- **No archive deletion or local data deletion without separate explicit user
  approval after the hold period.**
- Chroma is read-only throughout validation and dual-run.
- The frozen Chroma snapshot is immutable; only the disposable rollback delta
  is rebuilt from retained packages. A missing or checksum-invalid package
  blocks acknowledgement or rollback rather than silently losing the source.
- A failed gate returns to the preceding state; it never advances by waiver
  hidden in code.
- Reversing the GUI/no-GUI choice requires a new recorded decision; it does not
  roll back the successful data cutover or silently remove the handoff record.
- Refuse destructive paths resolving to a drive root, home directory,
  repository root or an unverified snapshot.
- Never restore an unsanitized secret-containing git ref to a public remote.

## Commit boundary

Use separate commits for:

1. state schema, runbook and tests;
2. read-only dual-run evidence integration;
3. supported-runtime Chroma dependency removal after the hold;
4. post-cutover GUI/no-GUI handoff record.

Do not include database deletion, history rewriting or broad layout cleanup.
