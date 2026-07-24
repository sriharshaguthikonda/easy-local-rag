# Issue #15: Atomically write vault JSON

[Roadmap ledger](README.md)

**Status:** PLANNING — the [decision-complete JIT packet](../superpowers/plans/2026-07-24-issue-015-atomic-vault-write-implementation.md) is under review; no code work is active.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/15
**Labels / priority:** `priority:P0`, `type:bug`
**Dependencies:** #8 remains open. This issue may use its already-landed `EASY_RAG_VAULT_SOURCE_DIR`/folder-picker fallback but does not satisfy or close #8. Generated vault data remains untracked.

## Implementation slices

1. Load one valid JSON list and merge by stable normalized absolute file identity. Treat `modification_time` and the additive `content_hash` as versions of that identity, not merge-key components, then write a same-directory temporary file followed by `os.replace`.
2. On corrupt input, preserve a timestamped backup before rebuilding; remove old duplicate append and mistaken backup branches.
3. Complete functional tests for unchanged repeat run (no duplicate), the same file identity with changed modification time/content hash replacing exactly once, invalid input backup, and injected write/replace failure preserving the target.

## Affected interfaces, files, and artifacts

- Worker-owned files are `vault_store.py`, `Vault_json_creation_from_HTMLs.py`, and `tests/test_vault_store.py`; `.gitignore` already ignores generated JSON and is verification-only.
- Artifact contract: `vault.json` is exactly one JSON array; no generated vault is committed.

## Concrete actions

- Keep helpers small and pure; reuse `os.replace` instead of a new storage layer.
- Ensure temporary files are cleaned on failure and a failed write cannot truncate the existing vault.
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
- **Commit:** `fix(#15): atomic-write vault json`.
