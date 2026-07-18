# Issue #14: Render inline source citations

**Status:** CLOSED on GitHub — live code has numbered prompt blocks and missing-citation UI warning; retain this plan as closure evidence and verify before archival.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/14
**Labels / priority:** `priority:P1`, `type:enhancement`
**Dependencies:** #5 guarded context formatter; #4 safe file-open behavior when links/actions are exposed.

## Implementation slices

1. Assign stable IDs to the exact sources sent to the model and include numbered context blocks plus a citation instruction.
2. Store source mapping with the answer, render only cited IDs that exist, and warn when the response returns no citation.
3. Keep source paths/actions validated; do not invent or post-process unsupported citations.

## Affected interfaces, files, and artifacts

- `streamlit_app.py`, `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `rag_prompting.py`, citation tests.
- Answer contract: context-backed claims use `[N]`; rendered footnotes map only to prompt-supplied sources.

## Concrete actions

- Verify source numbering starts at one, survives prompt assembly, and footnotes use the same map.
- Regression-test missing citations, invalid IDs, and two-source ordering.
- Verify manual output from two retrieved chunks before treating GitHub closure as implementation closure.

## Verification

```powershell
python -m pytest tests/test_streamlit_citations_config.py tests/test_rag_prompting.py -q
python -m pytest tests -q
python -m py_compile streamlit_app.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

## Closure gate, rollback, and commit boundary

- **Close/archive only when:** numbered context, inline citations, matching valid footnotes, and missing-citation warning are all demonstrated.
- **Rollback constraint:** keep the source expander and source mapping; do not show links for unvalidated paths.
- **Commit:** `feat(#14): render inline source citations`.
