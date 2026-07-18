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

### Early #26 phase — 26A and 26B

Runs after:

1. #2 rotates/revokes exposed credentials and identifies affected refs;
2. #18 inventories every branch, PR and local-only SHA.

It runs before #19 so new migration work is not built on tracked runtime debris
or unsafe public history. Its handoff gate closes #2, but does not close #26.

### Late #26 phase — 26C and 26D

Runs after #25 reaches `post_cutover_handoff`, whose immutable record names the
future GUI issue or explicit no-GUI decision. By then #21/#22 have frozen the
supported dependency surface and Chroma is no longer a supported runtime
dependency. It must not remove a migration tool, branch, snapshot or rollback
artifact needed by #19, #23 or #25.

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
- Use `git-filter-repo` 2.47 or later with `--sensitive-data-removal`. Record
  `First Changed Commit(s)`, affected `refs/pull/*/head` entries from
  `.git/filter-repo/changed-refs`, and any orphaned-LFS notice.
- Push all writable rewritten refs. Read-only `refs/pull/*` failures are
  expected and cannot be treated as remediated by force-push.
- If any PR ref or GitHub cached view retains an affected commit, open a GitHub
  Support sensitive-data-removal request with repository identity, affected PR
  count/numbers, first-changed commits, and any orphaned-LFS artifact. Wait for
  Support's written disposition.
- If Support removes the references, record confirmation and require a clean
  scan. If Support declines specifically because rotation/revocation mitigated
  the risk, map each remaining finding to read-only PR/cached refs, prove no
  writable ref reaches it, and request explicit user residual-risk acceptance.
  Any other refusal or missing acceptance permanently blocks #2/#19.
- Re-scan rewritten history with the same pinned command/config and keep
  affected secret types/ref names in a private operator record.
- Sanitized public archive refs may be created only after the clean outcome or
  the explicitly accepted immutable-residual gate; no archive may reach a
  residual affected commit.

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
- post-rewrite security scan report;
- `git-filter-repo` changed-refs/first-changed-commit record and, when needed,
  GitHub Support ticket/disposition without secret values;
- conditional hashed residual-ref/finding inventory and explicit user
  residual-risk acceptance;
- old-clone invalidation notice;
- #25 retention/rollback inventory.

## Concrete actions

1. Capture the #18 ref inventory and the current tracked-file inventory.
2. Rotate/revoke every affected credential under #2.
3. Repair ignore rules and untrack generated/private artifacts without deleting
   local copies.
4. Add secret and generated-file checks.
5. Coordinate and execute the sole approved history rewrite; force-push all
   writable refs once and record expected read-only PR-ref failures.
6. Obtain the conditional GitHub Support disposition for affected PR
   refs/cached views, apply the clean or accepted-residual terminal branch,
   then re-clone, re-fetch every remaining ref, scan, and invalidate old clones.
7. Continue migration work through #19–#25.
8. After #25, document supported commands and isolate legacy code.
9. Remove Chroma from core dependencies and prove a clean clone/install.
10. Record final repository size, tracked files and retained rollback artifacts.

## Verification

### Early phase verification

```powershell
git ls-files | rg "(__pycache__|\.pyc$|error\.log$|output\.log$|chroma\.sqlite3$|vault\.txt$)"
git check-ignore -v .env chroma/chroma.sqlite3 output.log __pycache__/example.pyc

$phase = 'pre-rewrite' # run post-rewrite only in a fresh clone after any required Support confirmation
$evidence = $env:EASY_RAG_SECURITY_EVIDENCE_DIR
if (-not $evidence) { throw 'Set EASY_RAG_SECURITY_EVIDENCE_DIR to an access-controlled directory outside the repository.' }
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*' '+refs/tags/*:refs/tags/*' '+refs/pull/*/head:refs/remotes/origin/pr/*'
git for-each-ref --format='%(refname) %(objectname)' refs/heads refs/remotes refs/tags |
  Set-Content (Join-Path $evidence "$phase-all-refs.txt")
gitleaks version | Set-Content (Join-Path $evidence "$phase-gitleaks-version.txt")
Get-FileHash .gitleaks.toml -Algorithm SHA256 |
  Format-List | Out-File (Join-Path $evidence "$phase-gitleaks-config.sha256")
$report = Join-Path $evidence "$phase-gitleaks.json"
gitleaks git --redact --config .gitleaks.toml --log-opts='--all' --report-format json `
  --report-path $report --exit-code 3 .
$scanExit = $LASTEXITCODE
$scanExit | Set-Content (Join-Path $evidence "$phase-gitleaks.exit-code")
if (-not (Test-Path $report)) { throw 'Gitleaks did not produce the required report.' }
$findings = @(Get-Content $report -Raw | ConvertFrom-Json)
Get-FileHash $report -Algorithm SHA256 |
  Format-List | Out-File (Join-Path $evidence "$phase-gitleaks-report.sha256")
if ($scanExit -notin 0,3) { throw "Gitleaks $phase scanner/configuration failure." }
if (($scanExit -eq 0) -ne ($findings.Count -eq 0)) { throw "Gitleaks $phase exit/report mismatch." }
if ($phase -eq 'post-rewrite' -and $scanExit -eq 3) {
  Write-Warning 'Post-rewrite findings require the Support-declined residual-risk branch; do not advance automatically.'
}
git diff --check
```

### Late phase verification

```powershell
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
after rewriting, using the same pinned Gitleaks version and config hash. When
`.git/filter-repo/changed-refs` names `refs/pull/*` or affected cached views
remain, finish the
[GitHub Support removal procedure](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
before the post-rewrite run. Run that final block in a fresh verification clone
so stale local PR refs cannot mask GitHub state. The private operator record
contains both ref manifests, redacted JSON reports, scanner/config/report
hashes, exit statuses, first-changed commits, affected PR numbers, and the
conditional Support disposition. Pre-rewrite exit `3` with a non-empty report
is expected findings evidence; exits other than `0`/`3`, a missing/malformed
report, or an exit/report mismatch are scanner failures.

The preferred acceptance result is fresh-clone post-rewrite exit `0` and zero
findings. A post-rewrite exit `3` is terminal only when Support's written
decline says rotation/revocation mitigated the risk, the report exactly equals
a dated/hashed residual inventory, reachability evidence proves every finding
exists only in read-only PR/cached refs and no writable ref, and the user
explicitly accepts that residual risk. Do not baseline or allowlist those
findings. Any mismatch, other refusal, or missing approval blocks #2/#19.

## Measurable closure gates

### Early phase handoff gate — 26A and 26B

- Every credential found in any ref is revoked/rotated before rewrite.
- The post-rewrite evidence reconciles the complete #18 manifest and proves
  every writable ref scans clean with the pinned scanner/config.
- Every affected read-only PR ref/cached view has a written GitHub Support
  disposition. The gate records either removal plus all-ref exit `0`, or the
  narrowly allowed Support-declined residual inventory/reachability evidence
  plus explicit user risk acceptance. If none were affected, the
  `changed-refs`/inventory evidence records that fact.
- `git ls-files` reports no interpreter cache, runtime log, local database,
  embedding, source corpus or private exported session.
- Ignore and secret/generated-file checks prevent recurrence.
- The old-clone invalidation notice and access-controlled recovery bundle are
  recorded.
- Passing this gate supplies #2 final-closure evidence and unlocks #19; #26
  remains open.

### Final issue closure gate — 26C and 26D

- A fresh clone contains only safe source/docs/fixtures and installs from the
  documented dependency set.
- The full focused test suite and supported CLI smoke pass in the clean clone.
- README names the supported, migration, experimental and legacy entry points
  separately.
- Every legacy file retained has purpose, defect, environment, replacement and
  deprecation status.
- Final tracked-file inventory and repository-size report are reviewed.
- #25 rollback artifacts remain checksum-valid and available.
- The post-#25 GUI implementation or explicit no-GUI decision is linked.

## Rollback and safety constraints

- Do not rewrite history before credential rotation and #18 inventory.
- Do not delete user-local data while untracking generated files.
- Keep an offline, access-controlled pre-rewrite recovery bundle; never push
  unsanitized history back to a public remote.
- Do not claim read-only PR refs or cached views were remediated by force-push.
  Support removal is preferred; the only alternative is the explicit,
  revocation-mitigated residual-risk branch above.
- Do not delete branches, migration exports or Chroma snapshots required by
  #19/#23/#25.
- Late layout moves are reversible commits separate from the history rewrite.
- Do not combine cleanup with functional retrieval/provider changes.

## Commit boundary

Keep these boundaries separate:

1. 26A early ignore/generated-file hygiene;
2. 26A CI/pre-commit safety checks;
3. 26B the sole coordinated history rewrite and public ref replacement;
4. 26C supported dependency/docs cleanup after #25 and the GUI decision;
5. 26D late legacy/layout/Chroma retirement after #25 and the GUI decision.

The early handoff and late final phase use separate PRs/commits. No bulk commit
mixes history rewriting, layout moves and runtime behaviour changes.
