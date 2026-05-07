# Issue #14 Plan: Render inline citations tied to source chunks

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/14

Priority: P1 enhancement

## Goal

Every answer claim that depends on retrieved context should cite source chunks
inline as `[1]`, `[2]`, etc. The existing collapsed source expander is not
enough for clinical or medical RAG use.

## Files to inspect

- `streamlit_app.py`
- `streamlit_groq_lama_chromadb_RAG_ETTS.py`
- prompt helper from issue #5 if it exists
- optional new file: `citations.py`
- tests under `tests/`

## Implementation steps

1. Reuse the context formatting helper from issue #5 if it exists.
2. Assign stable source numbers before calling the model:

   ```python
   numbered_sources = [
       {"id": 1, "file_name": ..., "text": ...},
       ...
   ]
   ```

3. Format context blocks like:

   ```text
   [1] file.html
   chunk text...

   [2] other.pdf
   chunk text...
   ```

4. Add a model instruction:

   ```text
   Cite the source number in square brackets after every factual claim that uses
   retrieved context. If context does not support a claim, say so.
   ```

5. Store `numbered_sources` in `st.session_state.current_sources` or a local
   response object.
6. Render the assistant response as Markdown so `[1]` remains visible.
7. Below the answer, render a compact footnote list:

   ```text
   [1] file.html - Open File - excerpt
   ```

8. Make source numbers clickable if safe path validation from issue #4 is
   already present. Otherwise show the path and use the existing safe open
   button.
9. Add post-processing only for display, not to invent citations. If the model
   omits citations, show a warning such as `No citations returned`.
10. Do not cite sources that were not sent to the model.

## Tests and verification

Add tests:

- source numbering is stable and starts at 1
- prompt contains `[1]` and `[2]`
- footnote renderer only renders cited IDs that exist
- missing citation warning appears when answer has no `[N]`

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile citations.py streamlit_app.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

Manual smoke:

1. Ask a question that is answered from two source chunks.
2. Expected: answer includes `[1]` or `[2]` next to claims.
3. Expected: source expander/footnotes show matching files and excerpts.

## Acceptance checklist

- [ ] Context sent to model is numbered.
- [ ] Prompt requires citations for context-backed claims.
- [ ] UI renders inline citations and matching footnotes.
- [ ] UI warns when citations are missing.
- [ ] Tests cover numbering and rendering.

## Commit boundary

Use one commit for this issue only:

```text
feat(#14): render inline source citations
```
