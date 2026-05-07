# Issue #8 Plan: Replace hardcoded Windows-only paths with config

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/8

Priority: P1 bug

## Goal

Remove user-specific paths such as `C:\Users\deletable\...` from runtime code.
The scripts should read paths from environment variables, config, or a folder
picker, with safe defaults based on `Path.home()`.

## Files to inspect

- `monitor_file_changes_update_chromaDB.py`
- `Vault_json_creation_from_HTMLs.py`
- `config.yaml`
- `.env.example`
- optional new file: `rag_config.py`
- tests under `tests/`

## Implementation steps

1. Add a small config helper, for example `rag_config.py`.
2. In the helper, define env var names:

   - `EASY_RAG_NLTK_DATA`
   - `EASY_RAG_MONITOR_DIR`
   - `EASY_RAG_VAULT_SOURCE_DIR`
   - `EASY_RAG_CHROMA_DIR`

3. Add a function:

   ```python
   def get_path(name: str, default: Path | None = None) -> Path:
       ...
   ```

   It should read the env var, expand `~`, and return an absolute `Path`.

4. In `monitor_file_changes_update_chromaDB.py`, replace the hardcoded NLTK path
   with:

   ```python
   nltk_data = get_path("EASY_RAG_NLTK_DATA", None)
   if nltk_data:
       nltk.data.path.append(str(nltk_data))
   ```

5. Replace the hardcoded monitor folder with:

   ```python
   folder = get_path("EASY_RAG_MONITOR_DIR", Path.home() / "Google Drive")
   monitor_folder(str(folder))
   ```

6. In `Vault_json_creation_from_HTMLs.py`, replace any hardcoded source path
   with `EASY_RAG_VAULT_SOURCE_DIR`. Keep the Tk folder picker as fallback if no
   env var is set.
7. Update `.env.example` with placeholders only. Do not commit real local paths
   unless they are comments showing examples.
8. Document the env vars in `README.md` or `AGENTS.md`.

## Tests and verification

Add tests for the helper:

- env var overrides default
- `~` expands
- missing optional path returns `None`
- missing required path raises a clear error

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile rag_config.py monitor_file_changes_update_chromaDB.py Vault_json_creation_from_HTMLs.py
```

Manual smoke:

```powershell
$env:EASY_RAG_MONITOR_DIR = "$PWD"
python monitor_file_changes_update_chromaDB.py
```

Stop after confirming it watches the configured folder, not a hardcoded user
folder.

## Acceptance checklist

- [ ] No runtime code hardcodes `C:\Users\deletable`.
- [ ] Monitor folder can be set by env var.
- [ ] Vault source folder can be set by env var or picker.
- [ ] NLTK data path is optional and configurable.
- [ ] `.env.example` documents path variables without real secrets.

## Commit boundary

Use one commit for this issue only:

```text
fix(#8): make runtime paths configurable
```
