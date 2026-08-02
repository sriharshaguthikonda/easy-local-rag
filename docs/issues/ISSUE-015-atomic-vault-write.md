# Issue #15: Atomically write vault JSON

[Roadmap ledger](README.md)

**Status:** CLOSED — COMPLETED on GitHub through [PR #36](https://github.com/sriharshaguthikonda/easy-local-rag/pull/36), merged into `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` as `88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205`. Its reviewed code head was `3ced61bcb9957907cc568e2e2560edd6f3c53b83`.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/15
**Labels / priority:** `priority:P0`, `type:bug`
**Dependencies:** #8 is closed independently. This issue used the already-landed `EASY_RAG_VAULT_SOURCE_DIR`/folder-picker fallback without being evidence for #8. Generated vault data remains untracked.

## Implementation slices

1. Validate one JSON list and merge by stable normalized absolute file identity without mutating either input. Treat `modification_time` and the additive `content_hash` as versions of that identity, not merge-key components. On replacement or hash upgrade, incoming canonical fields/chunks win, existing-only unknown entry keys survive, and incoming unknown-key collisions win; a same-version keep returns the existing object/value unchanged. At the same mtime, an existing hash plus an incoming missing hash is a keep: retain that exact existing hashed entry object at its original position, ignore every incoming canonical/chunk/unknown-key change, and preserve both inputs.
2. Validate before side effects, always create missing parent directories, then write one same-directory temporary file in the exact order `dump -> flush -> fsync -> close -> chmod-if-existing -> replace`. Preserve the old target until `os.replace` succeeds.
3. On corrupt input, move the byte-exact source with `os.rename` to a unique sibling backup before rebuilding. A backup-move failure leaves the corrupt target unchanged; a successful backup is followed by a valid replacement.
4. Complete the packet's exactly 17 named regressions for unchanged repeat, changed mtime with unchanged/absent hash, same-mtime changed hash, legacy hash upgrade/no downgrade, schema/backward compatibility, corruption/rebuild and backup-move failure, same-directory/order/mode/serialization, injected dump/UTF-8 encode-write/flush/fsync/close/chmod/replace/cleanup failures, builder integration, side-effect-free import, and no append mode.

## Affected interfaces, files, and artifacts

- Worker-owned files are `vault_store.py`, `Vault_json_creation_from_HTMLs.py`, and `tests/test_vault_store.py`; `.gitignore` already ignores generated JSON and is verification-only.
- The implementation branch is exactly `codex/fix-issue-15`, created at `619365224f6db3770d3369fd84a295589699e513`; its code PR targets `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
- Artifact contract: `vault.json` is exactly one JSON array; no generated vault is committed.

## Concrete actions

- Keep helpers small and pure; reuse `os.replace` instead of a new storage layer.
- Preserve the original dump/fsync/chmod/replace exception and attempt temporary-file cleanup. If cleanup itself fails, record the residual temp path on the original exception; the old target remains authoritative and the residual may remain for operator cleanup.
- Treat invalid in-memory vault data as `ValueError`; legacy entries may omit `content_hash`, but invalid required fields, hashes, non-finite numbers, or non-JSON-compatible unknown values fail validation.
- Exercise the writer through the vault builder once with a tiny temporary HTML directory.

## Verification

```powershell
python -c "import sys; print(sys.version); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
python -m pytest tests/test_vault_store.py -q
python -m pytest tests -q
python -m py_compile Vault_json_creation_from_HTMLs.py vault_store.py
```

Python 3.11 or newer is mandatory; a failing version preflight stops before edits. Manual: run the builder twice over one file and parse `vault.json`; expect one valid array and one entry.

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** repeat unchanged run has no duplicate; the same stable file identity with changed time and content replaces once; corruption backs up; injected failure preserves the old target; no append mode remains.
- **Retirement closure (mutually exclusive):** remove the vault builder from maintained entry points and docs, prove it cannot write generated vault data, and document the maintained persistence/export replacement with atomic-write evidence.
- **Rollback constraint:** preserve the old vault until a successful `os.replace`; never revert to append writes.
- **Closure evidence:** the final reviewed packet SHA was `14df1b7fcea0775c56b6a774c4047c3f14e29c9d`; immutable documentation evidence was `61fdb3b30d19451fd47b38c618c8d34630b8f547`; reviewed code head was `3ced61bcb9957907cc568e2e2560edd6f3c53b83`; and PR #36 merged with GitHub Create-a-merge-commit as `88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205`.
- **Commits and merge:** exactly `test(#15): specify atomic vault behavior`, then `fix(#15): atomic-write vault json`; each accepted code-review finding uses its own `fix(#15): address accepted review finding` commit. These boundaries are fixed by the [owner decision](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15#issuecomment-5070864427). The code PR uses GitHub **Create a merge commit** only—never squash or rebase—so the exact test, production, and accepted review-fix child SHAs remain reachable for evidence and rollback.
- **Frozen-base stop guard:** immediately before implementation work and again before code merge, fetch the GUI target branch and stop if its remote head is not exactly `619365224f6db3770d3369fd84a295589699e513`.
