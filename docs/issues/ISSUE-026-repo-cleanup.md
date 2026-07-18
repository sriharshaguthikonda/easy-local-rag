# Issue #26 Plan: Repository hygiene, history and layout cleanup

Status: **Open — split into early hygiene and late retirement cleanup**

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/26

Parent epic: [#17](https://github.com/sriharshaguthikonda/easy-local-rag/issues/17)

Labels: `priority:P1`, `type:dx`

## Goal

Make a fresh clone safe and understandable without destroying salvage or
rollback evidence. Security/history hygiene happens early; Chroma dependency
removal, legacy isolation and layout changes happen only after cutover.

## Dependencies and sequence

### Early #26 hygiene

Runs after:

1. #2 rotates/revokes exposed credentials and identifies affected refs;
2. #18 inventories every branch, PR and local-only SHA.

It runs before #19 so new migration work is not built on tracked runtime debris
or an unsafe public history.

### Late #26 cleanup

Runs after #25 succeeds, the future GUI decision is recorded and Chroma is no
longer a supported runtime dependency. It must not remove a migration tool,
branch, snapshot or rollback artifact needed by #19, #23 or #25.

## Implementation slices

### 26A — Early generated/private artifact hygiene

- Inventory tracked and untracked caches, logs, databases, embeddings, corpora,
  exports, IDE files and temporary downloads.
- Replace personal absolute `.gitignore` entries with repository-relative
  patterns while preserving safe synthetic fixtures.
- Remove generated files from tracking without deleting the user's local data.
- Add checks preventing new caches, logs, databases and secret-shaped values
  from entering commits.

### 26B — Credential-history remediation

- Complete rotation before rewriting any ref.
- Consume #18's immutable ref manifest, then fetch and scan local heads, remote
  heads, tags and PR refs without printing secret values.
- Coordinate the rewrite window, force-push consequences and old-clone
  invalidation.
- Execute the single approved history rewrite and public-ref replacement; no
  other issue owns or performs this operation.
- Re-scan rewritten history with the same pinned command/config and keep
  affected secret types/ref names in a private operator record.
- Sanitized public archive refs may be created only after the clean scan.

### 26C — Supported surface and dependencies

- Document exactly one supported CLI plus migration/audit commands.
- Mark experimental UIs, old database scripts and provider-specific demos
  unsupported until replaced.
- Pin reproducible core dependencies and separate optional provider, speech,
  GUI and migration dependencies after #21/#22 define the maintained surface.
- Make a clean environment install and smoke command reproducible.

### 26D — Late legacy/layout cleanup

- Remove Chroma from core runtime requirements only after #25.
- Keep read-only export/rollback tooling in an optional legacy migration area
  for the documented retention period.
- Move only code with a known supported or legacy disposition; do not perform a
  speculative whole-repository `src/` reshuffle.
- Record each removed entry point and its replacement.

## Affected interfaces, files and artifacts

Early files:

- `.gitignore`
- `.env.example`
- secret/generated-file CI or pre-commit configuration
- tracked caches, logs and local database artifacts identified by `git ls-files`

Late files:

- `README.md`, `AGENTS.md`, `TODO.md`
- dependency and lock files
- `docs/legacy/README.md`
- supported CLI/package entry points from #21/#22
- Chroma/Milvus/GUI scripts with recorded dispositions

Operator artifacts:

- private secret/ref inventory without values;
- before/after tracked-file and repository-size reports;
- clean-history scan report;
- old-clone invalidation notice;
- #25 retention/rollback inventory.

## Concrete actions

1. Capture the #18 ref inventory and the current tracked-file inventory.
2. Rotate/revoke every affected credential under #2.
3. Repair ignore rules and untrack generated/private artifacts without deleting
   local copies.
4. Add secret and generated-file checks.
5. Coordinate and execute the sole approved history rewrite; force-push once.
6. Re-scan every rewritten ref and invalidate old clones.
7. Continue migration work through #19–#25.
8. After #25, document supported commands and isolate legacy code.
9. Remove Chroma from core dependencies and prove a clean clone/install.
10. Record final repository size, tracked files and retained rollback artifacts.

## Verification

```powershell
git ls-files | rg "(__pycache__|\.pyc$|error\.log$|output\.log$|chroma\.sqlite3$|vault\.txt$)"
git check-ignore -v .env chroma/chroma.sqlite3 output.log __pycache__/example.pyc

$phase = 'pre-rewrite' # repeat with 'post-rewrite' after ref replacement
$evidence = $env:EASY_RAG_SECURITY_EVIDENCE_DIR
if (-not $evidence) { throw 'Set EASY_RAG_SECURITY_EVIDENCE_DIR to an access-controlled directory outside the repository.' }
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*' '+refs/tags/*:refs/tags/*' '+refs/pull/*/head:refs/remotes/origin/pr/*'
git for-each-ref --format='%(refname) %(objectname)' refs/heads refs/remotes refs/tags |
  Set-Content (Join-Path $evidence "$phase-all-refs.txt")
gitleaks version | Set-Content (Join-Path $evidence "$phase-gitleaks-version.txt")
Get-FileHash .gitleaks.toml -Algorithm SHA256 |
  Format-List | Out-File (Join-Path $evidence "$phase-gitleaks-config.sha256")
gitleaks git --redact --config .gitleaks.toml --log-opts='--all' --report-format json `
  --report-path (Join-Path $evidence "$phase-gitleaks.json") --exit-code 1 .
$scanExit = $LASTEXITCODE
$scanExit | Set-Content (Join-Path $evidence "$phase-gitleaks.exit-code")
if ($scanExit -ne 0) { throw "All-ref $phase scan found a secret or failed." }
Get-FileHash (Join-Path $evidence "$phase-gitleaks.json") -Algorithm SHA256 |
  Format-List | Out-File (Join-Path $evidence "$phase-gitleaks-report.sha256")

python -m pytest tests -q
python -m py_compile localrag.py
python localrag.py status
python localrag.py search "clean clone smoke" --json
python -m pip check
git count-objects -vH
git diff --check
```

`.gitleaks.toml` is the committed configuration, extends the default rules, and
may allowlist only reviewed synthetic fixtures. No baseline or allowlist may
hide the known revoked credential type. Run the block once before and once
after rewriting, using the same pinned Gitleaks version and config hash. The
private operator record contains both ref manifests, redacted JSON reports,
scanner/config/report hashes and exit statuses. The acceptance result is
post-rewrite exit `0` and zero findings across every ref in the #18 manifest;
missing refs, scanner errors or findings all fail the gate.

## Measurable closure gate

- Every credential found in any ref is revoked/rotated before rewrite.
- The post-rewrite all-ref secret scan covers the complete #18 manifest, exits
  `0`, and reports zero known exposed credentials using the same pinned scanner
  version and config as the recorded pre-rewrite scan.
- `git ls-files` reports no interpreter cache, runtime log, local database,
  embedding, source corpus or private exported session.
- A fresh clone contains only safe source/docs/fixtures and installs from the
  documented dependency set.
- The full focused test suite and supported CLI smoke pass in the clean clone.
- README names the supported, migration, experimental and legacy entry points
  separately.
- Every legacy file retained has purpose, defect, environment, replacement and
  deprecation status.
- Final tracked-file inventory and repository-size report are reviewed.
- #25 rollback artifacts remain checksum-valid and available.

## Rollback and safety constraints

- Do not rewrite history before credential rotation and #18 inventory.
- Do not delete user-local data while untracking generated files.
- Keep an offline, access-controlled pre-rewrite recovery bundle; never push
  unsanitized history back to a public remote.
- Do not delete branches, migration exports or Chroma snapshots required by
  #19/#23/#25.
- Late layout moves are reversible commits separate from the history rewrite.
- Do not combine cleanup with functional retrieval/provider changes.

## Commit boundary

Keep these boundaries separate:

1. early ignore/generated-file hygiene;
2. CI/pre-commit safety checks;
3. the sole coordinated history rewrite and public ref replacement;
4. supported dependency/docs cleanup;
5. late legacy/layout/Chroma retirement after #25.

No single bulk commit should mix history rewriting, layout moves and runtime
behaviour changes.
