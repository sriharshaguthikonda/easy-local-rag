# Issue #15: Atomically write vault JSON

[Roadmap ledger](README.md)

**Status:** PLANNING — the [decision-complete JIT packet](../superpowers/plans/2026-07-24-issue-015-atomic-vault-write-implementation.md) is under review; no code work is active.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/15
**Labels / priority:** `priority:P0`, `type:bug`
**Dependencies:** #8 remains open. This issue may use its already-landed `EASY_RAG_VAULT_SOURCE_DIR`/folder-picker fallback but does not satisfy or close #8. Generated vault data remains untracked.

## Implementation slices

1. Validate one JSON list and merge by stable normalized absolute file identity without mutating either input. Treat `modification_time` and the additive `content_hash` as versions of that identity, not merge-key components. On replacement or hash upgrade, incoming canonical fields/chunks win, existing-only unknown entry keys survive, and incoming unknown-key collisions win; a same-version keep returns the existing object/value unchanged.
2. Validate before side effects, always create missing parent directories, then write one same-directory temporary file in the exact order `dump -> flush -> fsync -> close -> chmod-if-existing -> replace`. Preserve the old target until `os.replace` succeeds.
3. On corrupt input, move the byte-exact source with `os.rename` to a unique sibling backup before rebuilding. A backup-move failure leaves the corrupt target unchanged; a successful backup is followed by a valid replacement.
4. Complete the packet's named regressions for unchanged repeat, changed mtime with unchanged/absent hash, same-mtime changed hash, legacy hash upgrade/no downgrade, schema/backward compatibility, corruption/rebuild and backup-move failure, same-directory/order/mode/serialization, injected dump/fsync/chmod/replace/cleanup failures, builder integration, side-effect-free import, and no append mode.

## Affected interfaces, files, and artifacts

- Worker-owned files are `vault_store.py`, `Vault_json_creation_from_HTMLs.py`, and `tests/test_vault_store.py`; `.gitignore` already ignores generated JSON and is verification-only.
- Artifact contract: `vault.json` is exactly one JSON array; no generated vault is committed.

## Concrete actions

- Keep helpers small and pure; reuse `os.replace` instead of a new storage layer.
- Preserve the original dump/fsync/chmod/replace exception and attempt temporary-file cleanup. If cleanup itself fails, record the residual temp path on the original exception; the old target remains authoritative and the residual may remain for operator cleanup.
- Treat invalid in-memory vault data as `ValueError`; legacy entries may omit `content_hash`, but invalid required fields, hashes, non-finite numbers, or non-JSON-compatible unknown values fail validation.
- Exercise the writer through the vault builder once with a tiny temporary HTML directory.

## Verification

```powershell
python -m pytest tests/test_vault_store.py -q
python -m pytest tests -q
python -m py_compile Vault_json_creation_from_HTMLs.py vault_store.py
```

Manual: run the builder twice over one file and parse `vault.json`; expect one valid array and one entry.

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** repeat unchanged run has no duplicate; the same stable file identity with changed time and content replaces once; corruption backs up; injected failure preserves the old target; no append mode remains.
- **Retirement closure (mutually exclusive):** remove the vault builder from maintained entry points and docs, prove it cannot write generated vault data, and document the maintained persistence/export replacement with atomic-write evidence.
- **Rollback constraint:** preserve the old vault until a successful `os.replace`; never revert to append writes.
- **Activation:** after this planning packet merges, code work requires a separate merged activation PR whose URL and immutable merge SHA are recorded in closure evidence.
- **Commits:** exactly `test(#15): specify atomic vault behavior`, then `fix(#15): atomic-write vault json`; each accepted code-review finding uses its own `fix(#15): address accepted review finding` commit. These boundaries are fixed by the [owner decision](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15#issuecomment-5070864427) and are not squashed before review.
