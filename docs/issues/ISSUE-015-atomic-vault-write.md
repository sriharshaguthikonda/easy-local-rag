# Issue #15 Plan: Atomic-write vault.json

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/15

Priority: P0 bug

## Goal

Stop `Vault_json_creation_from_HTMLs.py` from appending duplicate JSON arrays
and corrupting `vault.json`. The script should read the existing list, merge new
entries, and atomically replace the file.

## Files to inspect

- `Vault_json_creation_from_HTMLs.py`
- `.gitignore`
- tests under `tests/`

## Implementation steps

1. Add pure helper functions in `Vault_json_creation_from_HTMLs.py` or a new
   `vault_store.py`:

   - `load_vault(path) -> list`
   - `merge_vault_entries(existing, new_entries) -> list`
   - `atomic_write_json(path, data) -> None`

2. `load_vault` behavior:

   - missing file returns `[]`
   - valid JSON list returns the list
   - invalid JSON moves the corrupt file to `vault.json.invalid.<timestamp>.bak`
     and returns `[]` only after warning the user

3. `merge_vault_entries` behavior:

   - key by normalized `file_name`
   - if a file already exists and `modification_time` is unchanged, keep old
     entry
   - if modification time changed, replace the entry
   - append new files

4. `atomic_write_json` behavior:

   - write to a temp file in the same directory
   - flush and close it
   - use `os.replace(temp, path)`
   - clean temp file on failure

5. Replace both append-mode write blocks with one call:

   ```python
   existing = load_vault("vault.json")
   merged = merge_vault_entries(existing, new_data)
   atomic_write_json("vault.json", merged)
   ```

6. Remove the incorrect `Backup_vault.json` branch that still writes to
   `vault.json`.
7. Keep `vault.json` ignored. Do not commit generated vault data.

## Tests and verification

Add tests using `tmp_path`:

- missing vault writes one valid JSON list
- second run with same file does not duplicate entries
- changed `modification_time` replaces old entry
- invalid existing vault is backed up and replaced
- write failure does not leave partial target file

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile Vault_json_creation_from_HTMLs.py
```

Manual smoke:

1. Run the vault builder on a tiny folder with one HTML file.
2. Run it again on the same folder.
3. Open `vault.json`.
4. Expected: valid single JSON array, no duplicate top-level arrays, no
   comma-separated JSON fragments.

## Acceptance checklist

- [ ] No append-mode writes to `vault.json`.
- [ ] Existing vault is parsed before writing.
- [ ] Writes are atomic with `os.replace`.
- [ ] Duplicate unchanged files are not appended.
- [ ] Corrupt existing vault is backed up before replacement.
- [ ] Tests cover repeat runs and invalid JSON.

## Commit boundary

Use one commit for this issue only:

```text
fix(#15): atomic-write vault json
```
