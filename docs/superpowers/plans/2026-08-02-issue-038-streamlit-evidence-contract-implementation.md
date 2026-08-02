# Issue #38 Streamlit Evidence Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve retrieved document evidence from Chroma retrieval through every maintained Streamlit prompt, and fail visibly before a chat-provider call when retrieval fails.

**Architecture:** Keep the existing `tuple[str, list[dict]]` retrieval API, but make the ordered source list authoritative and derive the joined string from it. Number, guard, validate, display, and send the same source objects; retrieval exceptions propagate to the Streamlit request boundary, which displays the error and returns without calling Groq or Ollama.

**Tech Stack:** Python 3.11+, pytest, Streamlit, ChromaDB, Groq, Ollama; no new dependencies.

## Global Constraints

- Canonical issue: [GitHub #38](https://github.com/sriharshaguthikonda/easy-local-rag/issues/38).
- Canonical contract: [stable main copy](https://github.com/sriharshaguthikonda/easy-local-rag/blob/main/docs/issues/ISSUE-038-streamlit-evidence-contract.md) and [immutable ledger-era copy](https://github.com/sriharshaguthikonda/easy-local-rag/blob/f87b4cfcff4351c8f6e1e73ef43a7bf99e7fd5f9/docs/issues/ISSUE-038-streamlit-evidence-contract.md).
- Roadmap ledger: [stable main copy](https://github.com/sriharshaguthikonda/easy-local-rag/blob/main/docs/issues/README.md) and [immutable ledger-era copy](https://github.com/sriharshaguthikonda/easy-local-rag/blob/f87b4cfcff4351c8f6e1e73ef43a7bf99e7fd5f9/docs/issues/README.md).
- Code review vehicle: [PR #42](https://github.com/sriharshaguthikonda/easy-local-rag/pull/42).
- Frozen implementation base: `88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205`, the live GUI branch tip when work began.
- Implementation branch: `codex/issue-038-evidence-contract`.
- Merge target: `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
- Do not merge `main` into this GUI lineage. Do not mutate Chroma collections or generated evidence.
- Do not commit `.env`, `.serena/`, `__pycache__/`, `*.pyc`, the dirty `chroma-viewer` submodule, or unrelated runtime files.
- #38 is the approved independent P0 exception after #37. It does not unblock or advance the #2 -> #18 -> #26 migration-security sequence.

---

## Maintained-path decision and ownership

The maintained Streamlit path is fixed in place; no entry point is retired by this packet. Existing retrieval limits, ranking, MMR selection, model choices, token budget, streaming, TTS, and source display behavior remain unchanged except for the evidence and failure contracts below.

| File | Owned behavior |
|---|---|
| `streamlit_groq_lama_chromadb_RAG_ETTS.py` | Preserve each selected result's `document`, metadata, score, and order at the retrieval return boundary. |
| `rag_prompting.py` | Define source numbering, document validation, joined-context derivation, guarded source blocks, and response-citation validation. |
| `streamlit_app.py` | Route all four modes through one evidence path; send that prompt to the model; validate and display the same source objects. |
| `context_retrieval.py` | Raise on retrieval failure instead of returning a plausible answer string. |
| `context_search_synonyms_mmr.py` | Raise on hybrid-search failure instead of returning a plausible answer string. |
| `groq_lama_chromadb_RAG_ETTS.py` | Raise on retrieval failure instead of returning a plausible answer string. |
| `tests/test_rag_prompting.py` | Exercise malformed sources, mapping, prompt guards, and unknown citation IDs. |
| `tests/test_streamlit_citations_config.py` | Functionally exercise every mode, joined/structured parity, retrieval-to-model delivery, visible errors, and zero provider calls. |
| `tests/test_streamlit_hybrid_embeddings.py` | Exercise retrieval mapping/order/score shape and raised retrieval failures. |

Current caller map:

```text
streamlit_groq_lama_chromadb_RAG_ETTS.get_relevant_context_hybrid
  -> streamlit_app.retrieve_numbered_sources
  -> streamlit_app.process_chat_mode
     -> Standard | Focused Search | Brain Dump | Summary
  -> streamlit_app.chat_with_model
  -> streamlit_app.main source display
```

The other three corrected retrieval helpers are legacy/standalone seams. Their callers receive a raised `RuntimeError`; none can mistake retrieval failure for evidence text.

## Locked interfaces and failure behavior

`get_relevant_context_hybrid(...)` retains this public shape:

```python
tuple[str, list[dict]]
```

Each returned source has this minimum schema; existing safe metadata remains on the same object:

```python
{
    "file_name": "source.txt",
    "document": "non-empty retrieved chunk text",
    "score": 0.91,
}
```

The interface rules are locked:

- Preserve selected-source order. `build_numbered_sources()` adds sequential one-based `citation_id` and basename `source_name` without dropping fields.
- `context_from_sources()` is exactly the ordered source documents joined with `"\n\n"`; `retrieve_numbered_sources()` rejects any mismatch with the legacy joined string.
- `_source_document()` rejects missing, non-string, empty, or whitespace-only evidence before `build_guarded_context_block()` can build a provider prompt.
- `build_guarded_context_block()` sanitizes closing source tags and emits one `<source id="N" file="...">` block per source.
- `validate_response_citations()` extracts numeric `[N]` citations and raises when any ID is absent from the sent source list.
- Retrieval exceptions at all four named helpers are chained as `RuntimeError("Retrieval failed; no model request was made")`.
- `chat_with_model()` catches retrieval/validation failure at the outer request boundary, calls `st.error(...)`, returns `(None, None, None)`, and does not invoke Groq or Ollama.
- No new I/O, configuration, dependency, collection mutation, timeout, retry, or ranking behavior is introduced.

## Completed implementation tasks and commit boundaries

### Task 1: Specify the evidence contract

**Files:** `tests/test_rag_prompting.py`, `tests/test_streamlit_citations_config.py`, `tests/test_streamlit_hybrid_embeddings.py`

- [x] Add functional tests for non-empty documents, two-source citation mapping, all four Streamlit modes, joined/structured parity, and invented citation rejection.
- [x] Replace the stale citation source-text grep with functional prompt construction.
- [x] Commit as `bace44b728e627eb2aeb00c2063df139d90ab2f8` (`test(#38): specify Streamlit evidence contract`).

### Task 2: Correct the mapping assertion without locking ranking internals

**File:** `tests/test_streamlit_hybrid_embeddings.py`

- [x] Assert stable `(file_name, document)` mapping and float score presence without hard-coding ranking scores owned outside #38.
- [x] Commit as `bea6674a093f102ea04a1e4d1bb4360ab9f46fa0` (`test(#38): assert stable source mapping`).

### Task 3: Preserve evidence through prompt construction

**Files:** `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `rag_prompting.py`, `streamlit_app.py`

- [x] Build returned sources from selected metadata plus `document` and `final_score` at the return boundary.
- [x] Derive the joined context from that returned source list.
- [x] Route all four modes through `retrieve_numbered_sources()` and the guarded source block.
- [x] Validate response citation IDs against the exact sent/displayed source list.
- [x] Commit as `6accb869325fcdb287c5a6304f92554b57c654fe` (`fix(#38): preserve retrieved evidence through prompts`).

### Task 4: Prove evidence reaches the provider request

**File:** `tests/test_streamlit_citations_config.py`

- [x] Execute `chat_with_model()` with a fake retrieval result and fake streaming Groq response.
- [x] Assert the unique sentinel occurs exactly once in the submitted user message and source ID 1 still maps to that document.
- [x] Commit as `035aa7aba5a6f001f6557c4b382224c6da5f3d8c` (`test(#38): prove evidence reaches model request`).

### Task 5: Remove plausible-answer retrieval fallbacks

**Files:** the four retrieval modules above plus `tests/test_streamlit_citations_config.py` and `tests/test_streamlit_hybrid_embeddings.py`

- [x] Replace all four `"Answer this yourself!"` returns with chained retrieval errors.
- [x] Prove the Streamlit request boundary displays retrieval failure and invokes neither model provider.
- [x] Prove the maintained backend raises instead of returning answer-like text.
- [x] Commit as `29858e1f9c39d427ba827a492082f15b28a50d55` (`fix(#38): fail closed when retrieval errors`).

## TDD commands and measured evidence

Pristine GUI-lineage baseline:

```powershell
.\easyrag\Scripts\python.exe -m pytest tests -q
```

Measured result: `5 failed, 140 passed, 1 warning`. Four failures were the pre-existing `tests/test_rag_gui_comprehensive.py` GUI-mock failures; the fifth was the stale #38 citation test.

Initial focused baseline before the production fix was `1 failed, 6 passed`. During the RED/test correction sequence, the focused evidence suite recorded `1 failed, 12 passed`; the sole failure showed the first test had over-specified an unrelated numeric score. Commit `bea6674...` corrected that assertion. Production GREEN after `6accb86...` was:

```powershell
python -m pytest -p no:cacheprovider tests/test_rag_prompting.py tests/test_streamlit_hybrid_embeddings.py tests/test_streamlit_citations_config.py -q
```

Measured result: `13 passed`. The model-request regression then exposed its own missing injected validator (`1 failed, 13 passed`) before the test harness was corrected; final evidence at `035aa7...` was `14 passed`.

Gap 1 used an explicit RED/GREEN cycle:

```powershell
.\easyrag\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_streamlit_hybrid_embeddings.py tests/test_streamlit_citations_config.py -q
```

RED before replacing the fallback: `1 failed, 8 passed`; `test_hybrid_retrieval_failure_raises_instead_of_returning_an_answer` reported `DID NOT RAISE`. GREEN after the four boundary changes: `9 passed`.

Final measured gates after `29858e1...`:

```powershell
.\easyrag\Scripts\python.exe -m pytest tests -q
.\easyrag\Scripts\python.exe -m py_compile streamlit_app.py rag_gui.py GUI_direct_search.py rag_prompting.py streamlit_groq_lama_chromadb_RAG_ETTS.py context_retrieval.py context_search_synonyms_mmr.py groq_lama_chromadb_RAG_ETTS.py
git grep -n -F "Answer this yourself!" -- "*.py"
```

Measured results: full suite `4 failed, 152 passed, 1 warning`; all four failures are the same frozen-baseline GUI-mock failures. Compilation passed. The forbidden-fallback grep returned no matches.

## Rollback, review, and closure gates

No data migration or Chroma mutation occurred, so rollback is code-only. Before merge, preserve this packet as audit evidence and revert runtime/test commits newest-to-oldest with:

```powershell
git revert 29858e1f9c39d427ba827a492082f15b28a50d55 035aa7aba5a6f001f6557c4b382224c6da5f3d8c 6accb869325fcdb287c5a6304f92554b57c654fe bea6674a093f102ea04a1e4d1bb4360ab9f46fa0 bace44b728e627eb2aeb00c2063df139d90ab2f8
```

After merge, revert the recorded PR #42 merge commit with parent 1; never reset or rewrite the shared GUI branch.

The implementation is not closure evidence by itself. PR #42 must remain open until all of these gates pass on one exact head:

- [ ] Exact-head code review records PASS or maps every accepted finding to its fix commit.
- [ ] Independent verification reruns the focused suite, full suite, compilation, forbidden-fallback grep, and frozen-target guard.
- [ ] The PR base remains the GUI branch and its frozen target remains descended from `88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205` without a `main` merge.
- [ ] GitHub merges with **Create a merge commit**; do not squash or rebase.
- [ ] The issue receives immutable packet, commit, test, review, verifier, PR, merge, and rollback evidence before it is closed as completed.

Until those gates pass, do not merge PR #42 or close issue #38.
