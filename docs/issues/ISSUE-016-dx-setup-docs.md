# Issue #16 Plan: Add .env.example, AGENTS.md, and complete requirements

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/16

Priority: P1 developer experience

## Goal

A fresh clone should have enough setup files to run the repo without guessing
dependencies, secrets, or entry points.

## Files to inspect

- `requirements.txt`
- `README.md`
- `AGENTS.md`
- `.env.example`
- `.gitignore`
- all tracked Python entry points

## Required entry points to document

- CLI RAG: `localrag.py`
- email RAG: `emailrag2.py`
- Streamlit UI: `streamlit_app.py`
- PyQt GUI: `rag_gui.py` and split modules
- ingestion: `Text_embeddings_to_chromadb_python.py`
- file monitor: `monitor_file_changes_update_chromaDB.py`
- vault builder: `Vault_json_creation_from_HTMLs.py`
- Chroma helpers/viewers: `list_chromadb_collections.py`, `query_chromadb_*`

## Implementation steps

1. Create `.env.example` with placeholders only:

   ```dotenv
   GROQ_API_KEY=
   EASY_RAG_CHROMA_DIR=
   EASY_RAG_MONITOR_DIR=
   EASY_RAG_VAULT_SOURCE_DIR=
   EASY_RAG_NLTK_DATA=
   ```

2. Confirm `.env` is ignored in `.gitignore`.
3. Update or create `AGENTS.md` with:

   - working branch
   - source of truth for GitHub issues
   - warning about issue #2 key rotation
   - entry point map
   - test commands
   - rule to avoid committing generated data, `.env`, pyc files, and Chroma DB

4. Regenerate `requirements.txt` from imports, not by dumping the whole local
   venv. Include direct runtime dependencies:

   - `beautifulsoup4`
   - `chromadb`
   - `gTTS`
   - `groq`
   - `lxml`
   - `matplotlib`
   - `nest_asyncio`
   - `networkx`
   - `nltk`
   - `numpy`
   - `ollama`
   - `pandas`
   - `plotly`
   - `pydub`
   - `pymilvus`
   - `PyPDF2`
   - `PyQt5`
   - `python-dotenv`
   - `pyvis`
   - `PyYAML`
   - `rank_bm25`
   - `requests`
   - `SpeechRecognition`
   - `streamlit`
   - `tiktoken` if issue #13 lands
   - `torch`
   - `tqdm`
   - `watchdog`
   - `wordcloud`

5. Pin direct dependencies with compatible lower bounds or exact versions based
   on the current working environment. Do not include local path packages or
   generated caches.
6. Add `requirements-dev.txt` if tests need dev-only tools such as `pytest`.
7. Update `README.md` with one-command setup:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -U pip
   python -m pip install -r requirements.txt -r requirements-dev.txt
   ```

8. Add a short "choose an entry point" section so agents know which script to
   run for each workflow.

## Tests and verification

Suggested commands:

```powershell
python -m pip install -r requirements.txt --dry-run
python -m pytest tests -q
python -m py_compile localrag.py streamlit_app.py rag_gui.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py
Get-ChildItem -Recurse -File -Include '*.pyc' | Select-Object -First 5
```

Expected result: no `.pyc` files are staged for commit.

Manual smoke:

1. Create a fresh venv.
2. Install requirements.
3. Run `python -m pytest tests -q`.
4. Run `python -m py_compile` on the main entry points.

## Acceptance checklist

- [ ] `.env.example` exists and contains placeholders only.
- [ ] `AGENTS.md` maps the repo entry points.
- [ ] `requirements.txt` includes every direct runtime import.
- [ ] dev-only tools are separated or clearly documented.
- [ ] README has setup and entry point commands.
- [ ] No generated data or pyc files are staged.

## Commit boundary

Use one commit for this issue only:

```text
chore(#16): document setup and requirements
```
