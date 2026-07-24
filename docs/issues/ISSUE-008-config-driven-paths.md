# Issue #8: Make runtime paths configurable

[Roadmap ledger](README.md)

**Status:** OPEN — partial helpers may exist; audit every runtime path before closure.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/8
**Labels / priority:** `priority:P1`, `type:bug`
**Dependencies:** #26A provides early hygiene; #26C supplies final supported-path proof. #16 owns the public `.env.example` and setup documentation.

## Implementation slices

1. Keep one minimal `Path`-based config helper for monitor, vault source, Chroma, and optional NLTK data paths.
2. Replace user-specific runtime literals in the monitor and vault writer; expand `~`, resolve defaults from `Path.home()`, and retain the existing picker fallback.
3. Test overrides, optional absence, and required-path error behavior; document names in #16's setup material.

## Affected interfaces, files, and artifacts

- `rag_config.py`, `monitor_file_changes_update_chromaDB.py`, `Vault_json_creation_from_HTMLs.py`, `.env.example`, documentation, tests.
- Environment contract: `EASY_RAG_NLTK_DATA`, `EASY_RAG_MONITOR_DIR`, `EASY_RAG_VAULT_SOURCE_DIR`, `EASY_RAG_CHROMA_DIR`.

## Concrete actions

- Do not add a second configuration system; reuse the helper at each named call site.
- Ensure a missing optional NLTK path does nothing, while a required execution path fails with an actionable message.
- Search runtime Python files for the historical user-specific prefix before closing.

## Verification

```powershell
python -m pytest tests -q
python -m py_compile rag_config.py monitor_file_changes_update_chromaDB.py Vault_json_creation_from_HTMLs.py
$env:EASY_RAG_MONITOR_DIR = "$PWD"; python monitor_file_changes_update_chromaDB.py
```

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** no maintained runtime contains a hardcoded user path, overrides drive monitor/vault behavior, and picker/default behavior is tested.
- **Retirement closure (mutually exclusive):** remove an affected utility from supported entry points and docs, prove its machine-specific path is unreachable, and document the configurable maintained replacement. A retired file must not remain advertised as runnable.
- **Rollback constraint:** preserve safe defaults and never restore a machine-specific path.
- **Commit:** `fix(#8): make runtime paths configurable`.
