# Issue #18: Branch and PR inventory before cleanup

[Roadmap ledger](README.md)

**Status:** REVIEW — Aug 2 inventory evidence is historical; public refs were
reverified 2026-11-24; GitHub issue remains open pending current local-ref evidence and closure.
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
archive creation, and the post-rewrite clean terminal gate or documented,
explicitly accepted immutable-ref residual outcome terminal gate.

## Closure gate

Every non-default and local-only branch has a SHA, comparison, classification,
and proposed disposition; the pre-rewrite all-ref manifest and redacted report
are retained privately with scanner/config/report hashes and exit status;
affected PR numbers/refs are recorded for #26B's conditional GitHub Support
handoff; PR #1 has a documented superseded disposition; and useful concepts
plus proposed archives have explicit destination/26B handoffs. No archive ref
is created.

## 2026-08-02 evidence checkpoint

- Public inventory/classification: `docs/plans/branch-inventory.md` at local
  checkpoint `711480a` (Step 1 manifest checkpoint `12f1f9b`); these checkpoint
  SHAs remain local provenance for the Aug 2 capture; they are not public refs.
- Private exact-ref manifest and redacted report are outside the repository
  under user-only ACLs. Public hashes: manifest
  `fd7ca3b8dae52277d38d9c44a135ceebd15d5033d6f8891572208d4fe21e4862`;
  report `fdb51ad92f27397327ac1d48160fc8760c38075338eb7695abcc6b2291a0f247`;
  scanner config
  `7d863dffa2d7d8520f76de1c8013eed10676b718a6c62a66616157c6c68abf86`.
- Gitleaks `8.30.1` exited `3` with 13 redacted findings; exit/count agree.
  Finding types, paths, historical commit IDs, and affected PR refs are public;
  values are not.
- All 31 non-default remote heads, the default remote head, four local-only
  heads, six local/remote name
  divergences, 15 PR heads, and zero tags have immutable recorded tips and
  proposed dispositions. PR #1 is closed/unmerged and routes only to #24.
- No archive, tag, branch, merge, force-push, PR close, deletion, or rewrite was
  performed.

## 2026-11-24 public-ref verification

- This publication check used GitHub public branch, pull-request, PR-head, and
  tag refs only.
- All 32 documented remote heads and SHAs, all 15 documented PR heads and SHAs,
  and the zero-tag count matched live public metadata; `main` remained
  `f87b4cfcff4351c8f6e1e73ef43a7bf99e7fd5f9`.
- The four local-only heads, six local/remote divergences, private manifest, and
  redacted report remain the Aug 2 historical local capture; they were not
  re-captured here.

## Final verification (Aug 2 historical)

- The Aug 2 historical gate checkpoint `6fd0516` and public
  inventory/classification `711480a` are local provenance; the public refs below
  are the live verification.
- `python -m pytest tests -q --ignore=tests/test_gui_settings.py
  --ignore=tests/test_rag_gui_comprehensive.py`: 145 passed.
- `python -m py_compile streamlit_app.py rag_gui.py GUI_direct_search.py` and
  `python scripts/validate_issue_ledger.py --repo
  sriharshaguthikonda/easy-local-rag`: passed.
- The unfiltered test command stops during collection because this environment
  lacks `PyQt5`; only `tests/test_gui_settings.py` and
  `tests/test_rag_gui_comprehensive.py` are affected.
- Initial versus closeout ref evidence differs only at the advancing local work
  branch. Remote heads, PR refs, tags, and all other local heads are unchanged;
  no `archive/*` ref exists.
- Independent public-only evidence-contract review returned PASS after receiving
  the concrete hashes, counts, classification/register coverage, and no-mutation
  evidence.

The Aug 2 local evidence checkpoint passed its scoped inventory and verification
checks. This publication candidate keeps #18 open pending docs-only integration
and public closure evidence; it is not a readiness signal for #26A/#26B and does
not authorize ref mutation.

## Rollback constraints

No branch/PR deletion, merge, force-push, ref rewrite, or archive publication
under #18. Keep refs needed for #19 export and #25 rollback. #26B may act only
from the recorded SHA, scan outcome, explicit approval, and recovery plan.

## Commit boundary

One docs/evidence commit may add the inventory, proposed archive register, and
PR disposition after the scan completes. Tag, branch, archive, force-push,
PR-close, or delete operations are outside #18.
