# Issue #16: Make fresh-clone setup runnable

**Status:** OPEN — setup documentation and dependency inventory need a fresh-environment proof.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/16
**Labels / priority:** `priority:P1`, `type:dx`
**Dependencies:** #2 defines secret handling, #8 path variables, and #13 adds `tiktoken` when its implementation lands.

## Implementation slices

1. Add a non-secret example-only `.env.example`, keep `.env` ignored, and document the issue #2 rotation warning.
2. Map the CLI, email, Streamlit, PyQt, ingest, monitor, vault, and Chroma viewer entry points in `README.md` / `AGENTS.md`.
3. Rebuild direct runtime requirements from tracked imports, place test-only tools in `requirements-dev.txt`, then validate in a fresh virtual environment.

## Affected interfaces, files, and artifacts

- `requirements.txt`, `requirements-dev.txt`, `README.md`, `AGENTS.md`, `.env.example`, `.gitignore`.
- Setup contract: documented PowerShell venv install and chosen entry point work without relying on local caches.

## Concrete actions

- Do not dump an entire local `pip freeze`; retain direct dependencies with compatible bounds and include `tiktoken` only with its shipped use.
- Never place secrets, local Chroma, vault output, or bytecode in setup artifacts.
- Cross-check every tracked Python entry point against the documentation.

## Verification

```powershell
python -m pip install -r requirements.txt --dry-run
python -m pytest tests -q
python -m py_compile localrag.py streamlit_app.py rag_gui.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py
```

Manual: create a fresh venv, install runtime plus dev requirements, run the listed tests and compile checks.

## Closure gate, rollback, and commit boundary

- **Close only when:** a clean venv installs declared dependencies, entry points and environment variables are accurately mapped, `.env.example` has placeholders only, and generated data is not staged.
- **Rollback constraint:** keep explicit `.env` ignore and avoid replacing the curated requirements list with machine-specific freeze output.
- **Commit:** `chore(#16): document setup and requirements`.
