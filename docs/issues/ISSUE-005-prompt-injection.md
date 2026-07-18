# Issue #5: Treat retrieved documents as untrusted prompt context

**Status:** OPEN — guard helper and Streamlit integration exist in the live tree, but every Groq/Ollama retrieval path must be audited before closure.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/5
**Labels / priority:** `priority:P1`, `type:security`
**Dependencies:** Reuse the issue #14 source-numbering path where available; do not depend on ingest-time destructive sanitization.

## Implementation slices

1. Centralize a context guard and escaped, delimited source-block formatter in `rag_prompting.py`.
2. Route Streamlit, `GUI_workers.py`, and direct legacy Groq/Ollama paths through it; place the guard in the system/prompt layer.
3. Add pure formatter tests and a manual malicious-chunk smoke test.

## Affected interfaces, files, and artifacts

- `rag_prompting.py`, `streamlit_app.py`, `GUI_workers.py`, `streamlit_groq_lama_chromadb_RAG_ETTS.py`, and tests.
- Prompt contract: retrieved text is evidence only; embedded instructions, role changes, tool requests, and secret requests are ignored.

## Concrete actions

- Preserve normal document text; neutralize only closing wrapper tags so source text cannot escape its container.
- Format stable source IDs and include the guard on every model request with retrieval context.
- Trace every caller that concatenates `relevant_context` with user input; replace direct concatenation rather than fixing one UI only.

## Verification

```powershell
python -m pytest tests -q
python -m py_compile rag_prompting.py streamlit_app.py GUI_workers.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

Manual: use a chunk containing an instruction override and a literal closing source tag; confirm it is still fenced and the answer does not comply with it.

## Closure gate, rollback, and commit boundary

- **Close only when:** all retrieval-backed model paths use guarded blocks, wrapper escaping and source order are tested, and the malicious-chunk smoke succeeds.
- **Rollback constraint:** do not remove the guard or revert to raw concatenation; compatibility rollback may retain the formatter while disabling only optional display features.
- **Commit:** `fix(#5): fence retrieved context against prompt injection`.
