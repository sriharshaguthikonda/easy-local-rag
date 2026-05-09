# Issue #17 Plan: Migrate Chroma ingest/query to Qwen3 Embedding 0.6B

Status: proposed migration plan

## Goal

Move the active ChromaDB stack from `mxbai-embed-large` to
`qwen3-embedding:0.6b` without mixing vector spaces.

The first implementation slice should create a new Qwen-named Chroma store and
update `monitor_file_changes_update_chromaDB.py` so monitored HTML files are
embedded into that store with Qwen.

## Verified facts

- Local Ollama has `qwen3-embedding:0.6b` installed.
- Local Ollama `/api/embeddings` returned 1024 dimensions for
  `qwen3-embedding:0.6b`.
- Local Ollama `/api/embeddings` returned 1024 dimensions for
  `mxbai-embed-large`.
- Official Qwen model card lists Qwen3-Embedding-0.6B as text embedding,
  0.6B parameters, 32k context length, and output dimensions up to 1024:
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Ollama lists `qwen3-embedding:0.6b` as 639 MB with a 32K context window:
  https://ollama.com/library/qwen3-embedding:0.6b
- Current `chroma/chroma.sqlite3` has existing collections including:
  `html_chunks`, `html_chunks_temp`, and `html_chunks_text_in_documents`.
- `html_chunks_text_in_documents` currently has dimension 1024 and about
  212k embeddings, but no `embedding_model` collection metadata.
- Current code has an embedding contract guard in `embedding_contract.py`; that
  guard should reject non-empty legacy collections with missing model metadata.

Use the full 0.6B embedding output. Do not truncate, project, round, or cap Qwen
vectors to 1000 dimensions. The active target is the full 1024-dimensional vector
returned by Ollama for `qwen3-embedding:0.6b`.

## Current code paths

- Ingest from CSV: `Text_embeddings_to_chromadb_python.py`
- Folder monitor ingest: `monitor_file_changes_update_chromaDB.py`
- Chunking: `Semantic_chunking.py`
- Shared model contract: `embedding_contract.py`
- GUI query/settings: `GUI_direct_search.py`, `GUI_context.py`,
  `GUI_settings.py`, `rag_gui.py`
- Streamlit query path: `streamlit_groq_lama_chromadb_RAG_ETTS.py`,
  `streamlit_app.py`
- Legacy CLI/email paths still hardcode `mxbai-embed-large` in several files.

## Target names

Use names that encode model and profile:

```text
EASY_RAG_CHROMA_PATH=chroma_qwen3_0_6b
EASY_RAG_COLLECTION_NAME=html_chunks_text_in_documents_qwen3_0_6b
EASY_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b
EASY_RAG_CHUNK_PROFILE=qwen3_long_v1
```

Do not reuse `html_chunks_text_in_documents` for Qwen. Same dimensions do not
make the vector spaces compatible.

## Sparse + dense recommendation

Use both, but do not put sparse vectors into Chroma in this migration.

Recommended first version:

- Dense: Qwen embeddings in the new Chroma collection.
- Sparse: full-corpus BM25 over the same collection documents.
- Fusion: reciprocal-rank fusion or existing weighted fusion after tests.

Reason:

- The repo already uses `rank_bm25`.
- Issue #10 already defines full-corpus BM25 as the desired sparse path.
- Chroma is fine as the dense vector store here.
- A separate persisted sparse sidecar can be added later if full-corpus BM25
  rebuild cost is too high.

Do not switch vector databases just to get sparse+dense unless Chroma query
latency or corpus size becomes the blocker. If it does, evaluate Qdrant or
Vespa as a separate storage migration.

## Context length plan

Qwen's 32k context window is useful, but do not jump straight to 32k chunks.
Large chunks can reduce retrieval precision.

Start with a configurable chunk profile:

```text
qwen3_long_v1:
- token-aware or sentence-aware splitting
- larger than current 800-900 character chunks
- target 1500-3000 tokens per chunk for long clinical/reference pages
- keep overlap configurable
- store chunk_profile in Chroma metadata
```

If token-aware splitting is not added in the first slice, keep the existing
chunker and only switch model/store first. Then add the long chunk profile as a
second commit with retrieval-quality tests.

## Implementation steps

1. Add central config constants.
   - Move model/store names out of scattered hardcoded strings.
   - Keep env overrides for model, collection, Chroma path, and chunk profile.
   - Preserve `embedding_contract.py` model mismatch behavior.

2. Create the Qwen Chroma store.
   - Use `chroma_qwen3_0_6b` as the local persistent path.
   - Use `html_chunks_text_in_documents_qwen3_0_6b` as the collection.
   - Set metadata before ingest:

   ```python
   {
       "embedding_model": "qwen3-embedding:0.6b",
       "embedding_dimension": 1024,
       "embedding_context_length": 32768,
       "chunk_profile": "qwen3_long_v1",
       "hnsw:space": "cosine",
   }
   ```

3. Update `monitor_file_changes_update_chromaDB.py`.
   - Use the Qwen config defaults.
   - Create/get the Qwen collection.
   - Generate Qwen embeddings for new chunks.
   - Validate returned embedding dimension is exactly 1024 before
     `collection.add`.
   - Do not trim the vector to 1000 dimensions or otherwise reduce it.
   - Write monitor vault/checkpoint output to a Qwen-specific file, for example
     `temp_vault_qwen3_0_6b.json`.
   - Keep old `mxbai` collection untouched for rollback.

4. Update query paths after monitor ingest works.
   - `GUI_settings.py`: default embedding model and collection name.
   - `rag_gui.py`: add `qwen3-embedding:0.6b` to the embedding model combo.
   - `GUI_direct_search.py` and `GUI_context.py`: query the Qwen collection and
     keep metadata mismatch checks.
   - Streamlit paths: replace local hardcoded `model` and `collection_name`.
   - Legacy CLI/email paths: either update or explicitly mark as legacy.

5. Add tests.
   - Config default test: Qwen model/store names are selected.
   - Contract test: mismatched mxbai collection still fails.
   - Dimension test: Qwen embedding dimension is checked before add.
   - Monitor unit test: collection metadata is created with Qwen model.
   - Query test: GUI direct search rejects old collection metadata.

6. Add runtime smoke tests.
   - `ollama list` includes `qwen3-embedding:0.6b`.
   - `Invoke-RestMethod http://localhost:11434/api/embeddings` returns
     1024 dimensions.
   - Start monitor against a temp folder with two small HTML files.
   - Confirm new collection exists in `chroma_qwen3_0_6b/chroma.sqlite3`.
   - Confirm document count rises after file creation.
   - Query GUI/Streamlit using the Qwen collection.

## Verification commands

```powershell
ollama list
Invoke-RestMethod -Uri 'http://localhost:11434/api/embeddings' -Method Post -ContentType 'application/json' -Body '{"model":"qwen3-embedding:0.6b","prompt":"dimension check"}' | ForEach-Object { $_.embedding.Count }
python -m pytest tests -q
python -m py_compile embedding_contract.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py GUI_direct_search.py GUI_context.py GUI_settings.py streamlit_groq_lama_chromadb_RAG_ETTS.py streamlit_app.py
git diff --check
```

If system Python lacks dependencies, use the repo venv:

```powershell
.\easyrag\Scripts\python.exe -m pytest tests -q
.\easyrag\Scripts\python.exe -m py_compile embedding_contract.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py GUI_direct_search.py GUI_context.py GUI_settings.py streamlit_groq_lama_chromadb_RAG_ETTS.py streamlit_app.py
```

## Rollback

Rollback is env/config only if the old collection is left untouched:

```text
EASY_RAG_CHROMA_PATH=chroma
EASY_RAG_COLLECTION_NAME=html_chunks_text_in_documents
EASY_RAG_EMBEDDING_MODEL=mxbai-embed-large
```

Do not delete or relabel existing Chroma collections during the first Qwen
migration slice.

## Known blockers / notes

- `gh issue list` currently fails with `HTTP 401: Bad credentials`; public API
  still lists open issues.
- System Python does not have `chromadb`; the repo venv does.
- Importing Chroma through the repo venv timed out against the existing 13 GB
  `chroma/chroma.sqlite3`; direct SQLite inspection was used for collection
  names/counts.
- Existing generated/local files are dirty in the worktree. Stage only this
  plan unless deliberately taking over those changes.

## Acceptance checklist

- [ ] New Qwen Chroma path/collection is created.
- [ ] Monitor script writes only Qwen embeddings into the Qwen collection.
- [ ] Qwen collection metadata records model, dimension, context length, and
      chunk profile.
- [ ] Existing mxbai collection remains untouched.
- [ ] Query paths use the same Qwen model and collection by default.
- [ ] Sparse BM25 and dense Qwen retrieval run independently and fuse results.
- [ ] Verification commands pass or have documented environment blockers.

## Commit boundary

First commit:

```text
docs: plan qwen3 embedding migration
```

Implementation commits should be split after this plan is accepted.
