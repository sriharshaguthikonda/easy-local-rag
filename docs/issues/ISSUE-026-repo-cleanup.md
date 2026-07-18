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
- Scan branches, tags, PR refs and local refs without printing secret values.
- Coordinate the rewrite window, force-push consequences and old-clone
  invalidation.
- Re-scan rewritten history and keep affected secret types/ref names in a
  private operator record.
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
5. Coordinate and execute the history rewrite; force-push once.
6. Re-scan every rewritten ref and invalidate old clones.
7. Continue migration work through #19–#25.
8. After #25, document supported commands and isolate legacy code.
9. Remove Chroma from core dependencies and prove a clean clone/install.
10. Record final repository size, tracked files and retained rollback artifacts.

## Verification

```powershell
git ls-files | rg "(__pycache__|\.pyc$|error\.log$|output\.log$|chroma\.sqlite3$|vault\.txt$)"
git check-ignore -v .env chroma/chroma.sqlite3 output.log __pycache__/example.pyc
python -m pytest tests -q
python -m py_compile localrag.py
python localrag.py status
python localrag.py search "clean clone smoke" --json
python -m pip check
git count-objects -vH
git diff --check
```

The approved full-history secret scanner must run across all branches, tags and
PR refs before and after rewriting. Its command and sanitized output belong to
the private operator record because affected secret locations must not be
published casually.

## Measurable closure gate

- Every credential found in any ref is revoked/rotated before rewrite.
- The post-rewrite all-ref secret scan reports zero known exposed credentials.
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
3. coordinated history rewrite and public ref replacement;
4. supported dependency/docs cleanup;
5. late legacy/layout/Chroma retirement after #25.

No single bulk commit should mix history rewriting, layout moves and runtime
behaviour changes.
