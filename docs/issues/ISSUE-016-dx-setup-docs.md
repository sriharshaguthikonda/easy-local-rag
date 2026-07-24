# Issue #16: Make fresh-clone setup runnable

[Roadmap ledger](../plans/README.md)

**Status:** OPEN — setup documentation and dependency inventory need a fresh-environment proof.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/16
**Labels / priority:** `priority:P1`, `type:dx`
**Dependencies:** 16A depends on #2's secret-handling rules and may land immediately; 16B depends on #21/#22 defining the maintained client/provider surface, plus #8 path variables and #13's shipped `tiktoken` use where those paths remain supported.

## Implementation slices

### 16A — early secret-safe contributor setup

1. Add a placeholder-only `.env.example`, keep `.env` ignored, document the issue #2 rotation warning, and add the minimal PowerShell venv/test bootstrap.
2. Verify no credential-shaped value or generated local data is present or staged.

### 16B — maintained dependency and entry-point surface

1. After #21/#22 settle the maintained clients/providers, map only supported CLI, email, Streamlit, PyQt, ingest, monitor, vault, and Chroma viewer entry points in `README.md` / `AGENTS.md`; mark retired paths unsupported.
2. Rebuild direct runtime requirements from imports of maintained paths, place test-only tools in `requirements-dev.txt`, then validate in a fresh virtual environment.

## Affected interfaces, files, and artifacts

- `requirements.txt`, `requirements-dev.txt`, `README.md`, `AGENTS.md`, `.env.example`, `.gitignore`.
- Setup contract: documented PowerShell venv install and chosen entry point work without relying on local caches.

## Concrete actions

- Do not dump an entire local `pip freeze`; retain direct dependencies with compatible bounds and include `tiktoken` only with its shipped use.
- Never place secrets, local Chroma, vault output, or bytecode in setup artifacts.
- Cross-check every tracked Python entry point against the documentation.

## Verification

16A:

```powershell
git check-ignore .env
git grep -n -I -E '(api[_-]?key|token|secret)[[:space:]]*=[[:space:]]*[^<${]' -- .env.example README.md
python -m venv .venv-setup-smoke
```

16B:

```powershell
python -m pip install -r requirements.txt --dry-run
python -m pytest tests -q
python -m py_compile localrag.py streamlit_app.py rag_gui.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py
```

Manual: create a fresh venv, install runtime plus dev requirements, run the listed tests and compile checks.

## Closure gate, rollback, and commit boundary

- **16A gate:** `.env` is ignored, `.env.example` contains placeholders only, the rotation warning and bootstrap are documented, and no secret/generated artifact is staged.
- **16B maintained-path gate:** after #21/#22, a clean venv installs declared dependencies, maintained entry points and environment variables are accurately mapped, and their compile/tests pass.
- **16B retirement gate (mutually exclusive per entry point):** an unmaintained path is explicitly unsupported in docs, absent from the supported setup commands, and has a named maintained replacement with runnable evidence.
- **Rollback constraints:** never weaken `.env` ignore or secret guidance. Revert 16B dependency/doc mapping independently without removing 16A; never replace curated requirements with a machine-specific freeze.
- **Commits:** 16A `docs(#16): add secret-safe contributor setup`; 16B `chore(#16): document maintained setup and requirements`.
