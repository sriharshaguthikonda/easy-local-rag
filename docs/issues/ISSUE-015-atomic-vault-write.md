# Issue #15: Atomically write vault JSON

**Status:** OPEN — `vault_store.py` and basic integration exist; this is the next small closure after missing unchanged-entry and write-failure regressions.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/15
**Labels / priority:** `priority:P0`, `type:bug`
**Dependencies:** #8 supplies configurable source paths; generated vault data remains untracked.

## Implementation slices

1. Load one valid JSON list, merge entries by normalized `file_name` and `modification_time`, then write a same-directory temporary file followed by `os.replace`.
2. On corrupt input, preserve a timestamped backup before rebuilding; remove old duplicate append and mistaken backup branches.
3. Complete functional tests for unchanged repeat run (no duplicate), changed replacement, invalid input backup, and injected write/replace failure preserving the target.

## Affected interfaces, files, and artifacts

- `vault_store.py`, `Vault_json_creation_from_HTMLs.py`, `.gitignore`, `tests/test_vault_store.py`.
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

- **Close only when:** repeat unchanged run has no duplicate, changed input replaces once, corruption backs up, injected failure preserves the old target, and no append mode remains.
- **Rollback constraint:** preserve the old vault until a successful `os.replace`; never revert to append writes.
- **Commit:** `fix(#15): atomic-write vault json`.
