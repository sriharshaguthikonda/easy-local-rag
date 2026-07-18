# Issue #18: Branch and PR inventory before cleanup

**Status:** Open; early preservation/security gate.
**GitHub:** [#18](https://github.com/sriharshaguthikonda/easy-local-rag/issues/18)
**Parent epic:** #17.
**Labels / priority:** `priority:P0`, `type:security`.

## Dependencies

Blocked by #2 credential containment for any public archive publication. Runs after #2 begins and before early #26 hygiene, #19 export, or any branch/PR cleanup. It contributes the archive register required by #25 and late #26.

## Implementation slices

1. **Enumerate:** collect remote, local-only, tag, PR, and default refs; branch names, immutable tip SHAs, unique commits/files, entry points, database/provider assumptions, tests, generated files, and private paths.
2. **Classify:** record salvage candidates, broken behavior, credential findings by type/location (never values), and disposition: port, archive, close, or delete after verification.
3. **Preserve safely:** secret-scan all refs; create only sanitized immutable archive refs; register them in migration documentation.
4. **Dispose deliberately:** document PR #1's superseded status and focused destination issues before closing it; no deletion until #19 and #25 gates.

## Affected interfaces, files, and artifacts

- `docs/plans/branch-inventory.md` is the public inventory baseline: `main`, the BM25 experiment (`ed662570...`), the QtPy branch (`8cb080...`), PR #1, and closed-fix salvage commits.
- `docs/plans/issue-disposition.md` records destination issues and legacy closure rules.
- New evidence: local-ref appendix, all-ref secret-scan summary, sanitized archive-ref register, PR #1 disposition note, and generated-artifact classification.
- Git refs: `archive/easy-rag-main-before-postgres`, `archive/easy-rag-gui-experiments`, and `archive/easy-rag-streamlit-experiments` only after sanitization.

## Concrete actions

- Compare every non-default ref with `main`; record unique entry points, collections, provider use, fixtures, generated artifacts, and port targets.
- Inspect every local-only ref with `git show-ref`, then add its SHA/disposition before deletion is even proposed.
- Scan all refs for secrets; rotate/remediate under #2 before publishing or retaining a public archive containing affected history.
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
gitleaks git --redact --config .gitleaks.toml --log-opts='--all' --report-format json `
  --report-path (Join-Path $evidence 'pre-rewrite-gitleaks.json') --exit-code 1 .
if ($LASTEXITCODE -ne 0) { throw 'All-ref scan found a secret or failed; do not publish archive refs.' }
```

Record `gitleaks version`, the `.gitleaks.toml` SHA-256, command exit status, and
report SHA-256 beside the private report. The ref manifest must cover local
heads, remote heads, tags, and fetched PR refs. Exit `0` and zero findings are
required before an archive can be classified as sanitized; #26B owns any
rewrite and the matching post-rewrite scan.

## Closure gate

Every non-default and local-only branch has a SHA, comparison, classification, and disposition; the pre-rewrite all-ref manifest and redacted report are retained privately with scanner/config hashes and exit status; every retained archive has passed secret remediation and has an immutable recorded ref; PR #1 has a documented superseded disposition; and useful concepts have explicit destination issues.

## Rollback constraints

No branch/PR deletion, merge, force-push, or archive publication without the recorded SHA, scan outcome, and approved sanitized archive ref. Keep refs needed for #19 export and #25 rollback. Restoring a prematurely hidden ref means recreating it from its recorded SHA only after security review.

## Commit boundary

One docs/evidence commit may add the inventory, archive register, and PR disposition after scans pass. Tag, branch, PR-close, or delete operations are separate, explicitly approved GitHub/Git actions.
