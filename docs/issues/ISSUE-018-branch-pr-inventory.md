# Issue #18: Branch and PR inventory before cleanup

**Status:** Open; early preservation/security gate.
**GitHub:** [#18](https://github.com/sriharshaguthikonda/easy-local-rag/issues/18)
**Parent epic:** #17.
**Labels / priority:** `priority:P0`, `type:security`.

## Dependencies

Runs after #2's credential-containment gate and before early #26, #19 export,
or any branch/PR cleanup. It contributes an immutable ref inventory, scan
evidence, and proposed archive dispositions; #26B alone sanitizes/replaces refs
and creates any approved archive.

## Implementation slices

1. **Enumerate:** collect remote, local-only, tag, PR, and default refs; branch names, immutable tip SHAs, unique commits/files, entry points, database/provider assumptions, tests, generated files, and private paths.
2. **Classify:** record salvage candidates, broken behavior, credential findings
   by type/location (never values), affected PR numbers/refs, and disposition:
   port, archive, close, or delete after verification.
3. **Preserve safely:** secret-scan all refs, record immutable tips, and propose
   archive names/dispositions without creating or publishing archive refs.
4. **Dispose deliberately:** document PR #1's superseded status and focused destination issues before closing it; no deletion until #19 and #25 gates.

## Affected interfaces, files, and artifacts

- `docs/plans/branch-inventory.md` is the public inventory baseline: `main`, the BM25 experiment (`ed662570...`), the QtPy branch (`8cb080...`), PR #1, and closed-fix salvage commits.
- `docs/plans/issue-disposition.md` records destination issues and legacy closure rules.
- New evidence: local-ref appendix, all-ref secret-scan summary, proposed
  archive-ref register, PR #1 disposition note, and generated-artifact
  classification.
- Conditional Support handoff: affected PR numbers/refs and finding commit IDs,
  without secret values; #26B adds `git-filter-repo` first-changed commits.
- Proposed only: `archive/easy-rag-main-before-postgres`,
  `archive/easy-rag-gui-experiments`, and
  `archive/easy-rag-streamlit-experiments`. #18 does not create them.

## Concrete actions

- Compare every non-default ref with `main`; record unique entry points, collections, provider use, fixtures, generated artifacts, and port targets.
- Inspect every local-only ref with `git show-ref`, then add its SHA/disposition before deletion is even proposed.
- Scan all refs for secrets and hand the findings plus proposed archive
  dispositions to #26B; do not publish or retain an archive ref under #18.
- Preserve concepts only: worker/progress, streaming display, search-only, source/score views, status/reconnect, file-change concepts, and regression fixtures. Do not port direct Chroma/provider/UI coupling.
- Close PR #1 only after linking its QtPy wrapper disposition to future #24; never merge it wholesale.

## Verification

```powershell
git show-ref
git branch -a --no-merged main
gh pr view 1 --repo sriharshaguthikonda/easy-local-rag
gh issue view 18 --repo sriharshaguthikonda/easy-local-rag

$evidence = $env:EASY_RAG_SECURITY_EVIDENCE_DIR
if (-not $evidence) { throw 'Set EASY_RAG_SECURITY_EVIDENCE_DIR to an access-controlled directory outside the repository.' }
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*' '+refs/tags/*:refs/tags/*' '+refs/pull/*/head:refs/remotes/origin/pr/*'
git for-each-ref --format='%(refname) %(objectname)' refs/heads refs/remotes refs/tags |
  Set-Content (Join-Path $evidence 'pre-rewrite-all-refs.txt')
$report = Join-Path $evidence 'pre-rewrite-gitleaks.json'
gitleaks git --redact --config .gitleaks.toml --log-opts='--all' --report-format json `
  --report-path $report --exit-code 3 .
$scanExit = $LASTEXITCODE
$scanExit | Set-Content (Join-Path $evidence 'pre-rewrite-gitleaks.exit-code')
if (-not (Test-Path $report)) { throw 'Gitleaks did not produce the required report.' }
$findings = @(Get-Content $report -Raw | ConvertFrom-Json)
Get-FileHash $report -Algorithm SHA256 |
  Format-List | Out-File (Join-Path $evidence 'pre-rewrite-gitleaks-report.sha256')
if ($scanExit -notin 0,3) { throw 'Gitleaks scanner/configuration failure.' }
if (($scanExit -eq 0) -ne ($findings.Count -eq 0)) { throw 'Gitleaks exit/report mismatch.' }
```

Record `gitleaks version`, the `.gitleaks.toml` SHA-256, command exit status, and
report SHA-256 beside the private report. The ref manifest must cover local
heads, remote heads, tags, and fetched PR refs. Exit `3` with a non-empty,
redacted report is valid pre-rewrite inventory evidence and does not block #18
closure; any other nonzero exit is a scanner failure. #26B owns remediation,
archive creation, and the required exit-`0` post-rewrite scan.

## Closure gate

Every non-default and local-only branch has a SHA, comparison, classification,
and proposed disposition; the pre-rewrite all-ref manifest and redacted report
are retained privately with scanner/config/report hashes and exit status;
affected PR numbers/refs are recorded for #26B's conditional GitHub Support
handoff; PR #1 has a documented superseded disposition; and useful concepts
plus proposed archives have explicit destination/26B handoffs. No archive ref
is created.

## Rollback constraints

No branch/PR deletion, merge, force-push, ref rewrite, or archive publication
under #18. Keep refs needed for #19 export and #25 rollback. #26B may act only
from the recorded SHA, scan outcome, explicit approval, and recovery plan.

## Commit boundary

One docs/evidence commit may add the inventory, proposed archive register, and
PR disposition after the scan completes. Tag, branch, archive, force-push,
PR-close, or delete operations are outside #18.
