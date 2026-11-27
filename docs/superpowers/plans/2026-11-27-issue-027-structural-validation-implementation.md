# Issue #27 maintained-Chroma structural validation implementation packet

**State** Locked by the packet PR. A separate activation PR must merge before any product or RED-test edits.
**Mode** One-time maintained ChromaDB exception, deterministic extractive validation.
**Target** `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
**Frozen base** `656ae2a905cb4a400e66a9101663de546b2d17e0`.
**Owner decision, 2026-11-27** “I approve a separate, narrowly scoped maintained-Chroma implementation exception for the deterministic structural-validation portion of #27 while #2 remains open.” Approval covers the packet, separate activation, tests-only RED commits, offline implementation and independent verification. The owner requires all four maintained answer entry points and keeps full #27 open. [Issue #27](https://github.com/sriharshaguthikonda/easy-local-rag/issues/27) · [Canonical acceptance](../../issues/ISSUE-027-grounded-answers.md) · [Ledger](../../issues/README.md) · [Pinned audit](../../Research/issue-027/README.md).

## Callers and ownership

| Worker | Scope |
| --- | --- |
| A | Typed result and strict parser in `rag_prompting.py`, with parser cases in `tests/test_structural_answers.py`. |
| B | Current evidence snapshot, deterministic membership and quotation verification in `rag_prompting.py` and exact request/citation tests. |
| C | Quarantined completion orchestration in `rag_prompting.py`, terminal markers and buffer bounds in `generation_models.py`, provider and cancellation tests. |
| D | Qt `GUI_workers.py` / `GUI_chat.py` / `rag_gui.py`; Streamlit `streamlit_app.py` synchronous and worker paths; both CLI and programmatic wrappers in `groq_lama_chromadb_RAG_ETTS.py` and `streamlit_groq_lama_chromadb_RAG_ETTS.py`; update existing generation fixtures and add sink regressions. |
| E | Independent read-only Ornith review of the exact implementation head and synthetic verification logic. The coordinator reruns tests on an independent pinned snapshot and checks findings against the source. |

The maintained Chroma entry points are the same four named by the closed #5 activation. The historical vault-based `localrag.py`, Milvus scripts, rewrite routes and direct lexical search do not use this answer contract. Retain their existing scope.

## Locked contract

- Add a trusted instruction to the existing #5 composer. Accept only a single UTF-8 JSON object of the form `{"excerpts":[{"alias":1,"quote":"Exact source excerpt."}]}`. The object and records have exactly the listed keys. `alias` is a positive integer, with booleans rejected. Require one through 32 excerpts, each nonblank and at most 4,096 code points, and at most 65,536 UTF-8 bytes for the provider payload. Reject duplicate keys, duplicate normalized excerpt records, nonfinite values, unknown/missing keys, wrong types, malformed UTF-8 or surrogates, fences and trailing syntax. There is no free-text answer field or raw citation-marker path.
- Represent parsed excerpts as ordered frozen records. Carry a typed result with `structurally_validated`, `invalid_response`, `insufficient_evidence`, `input_budget`, `incomplete`, `cancelled`, `timeout` or `provider_failed`, with safe reason, actual provider/model metadata where available, a structurally accepted excerpt tuple only on success, and evidence snapshot/digests where available. Use the result's application-rendered text for compatible string-returning wrappers and existing callback interfaces. Failures expose zero answer text and offer the available evidence view. Reserve the word “grounded” for the full #27 contract.
- Compose a frozen request package from the ordered, numbered source rows included in the final guarded user message. Preserve the original source text, collection ID, chunk ID, locator and supplied metadata in the snapshot. Guard against duplicate aliases and duplicate stable pairs. Scope alias resolution to this turn. Recompute the guarded evidence block from the snapshot and verify the exact current message before router dispatch. Streamlit may trim history while retaining the identical system and current turns. If evidence does not fit, reject the entire request before answer dispatch. Keep SHA-256 of the exact guarded evidence block and a canonical JSON hash of the snapshot, with no corpus text in logs. Store accepted turn source/digest metadata separately from replayed text. Preserve explicitly retained identity components without inventing provenance for older programmatic callers.
- Validate each returned alias against the sent source map. Normalize source and quote with Unicode NFC and CRLF/CR-to-LF only. Match a contiguous occurrence with original case, spaces and punctuation under those rules. Render the matched source span from the snapshot with its application-owned numeric alias. This boundary accepts only extractive evidence. It requires no entailment judge and does not make a semantic claim, detect contradictions, estimate evidence sufficiency, or apply a medical profile.
- Keep output quarantined through the shared router and application boundary. Pass no raw-output callback to a sink. Require a normal `stop` terminal marker for answer requests and reject absent/unsupported terminal markers, length stops, buffer overflow, empty text, output after a terminal marker, interrupted streams, timeouts and late cancellation. Keep existing generic routing behavior for rewrites, same-provider pre-output fallback with the identical guarded request, strict pinning and text-only policy. Make no extra model call for repair. Return evidence only on unsuccessful status.
- Keep raw provider data out of display, history, export, status/error text and speech, including partial output. Commit accepted text/history only after the cancellation and validation boundary. Preserve current literal UI rendering and explicit cloud-speech consent. Maintain Qt and Streamlit search-only and existing CLI lexical search without generation, embedding, or speech. Explicit saved-chat export may read accepted display/history and must never capture provider packets. Current per-turn state retains the snapshot and stable IDs for source resolution.
- Rollback is evidence and offline search. A runtime or provider-incompatibility failure never falls back to unrestricted prose.

## Review and activation evidence

The ChatGPT advisory review required explicit Unicode reconstruction tests,
turn isolation and cancellation at publication, and terminal validation across
provider packet shapes and fallback. Cover repeated normalized substrings,
identical per-turn aliases paired with different collection/chunk IDs, mutation
of input rows after snapshot creation, a cancellation at the last publication
boundary, and failed termination after complete-looking JSON. Keep any usage
metadata after a normal stop from affecting the rendered answer. Check that
approved pre-output fallback receives identical current evidence and obtains an
empty buffer for the replacement attempt.

The activation PR must link to the merged packet containing the recorded owner
approval and bind both the packet and activation to the exact maintained GUI
base. The #27 issue stays open throughout. The maintained product branch cannot
change before activation merges.

## Test and commit gates

1. After activation, exercise public call paths on the frozen tree. Commit the application-level structural RED cases while product files still match the pin. Reproduce unknown and composite markers, absent sent-set aliases with a distinct excluded document sentinel, fabricated or wrong-source quotes, JSON/type failures, empty completions and length/unterminated streams. Record assertion failures with text leaking into callback/display/history/export where documented. Add or isolate a second tests-only RED commit for pending chunks, cancellation, late errors and speech/export boundaries. Controls cover existing empty retrieval, request budget, guarded roles, text-only policy and provider-free search. Tests target real composer/router/application paths using fake clients and synthetic source records.
2. Implement the strict parser and frozen sent-evidence contract. Wire the router's completeness-only path and the four entry points. Update existing mocks to provide completed structured packets and assert their accepted source text and stable IDs. Exercise distinct same-basename sources, identical chunk names in different collections, Unicode/line endings, alias reuse, budget loss and cancellation after the last received content. No cross-provider fallback, no raw or partial output callback into any sink.
3. Run the offline verification commands. Verify the guard denies loopback TCP and DNS before pytest. Use Python 3.12, ChromaDB 1.0.16, Qt offscreen, Chroma telemetry disabled, no provider credentials in the test process and synthetic temporary Chroma collections. Review the exact candidate SHA independently before any integration. An unapproved live model smoke is outside this packet, including the prior #5 approval.
4. Record RED, focused, full and native outcomes and the independent review. Recheck the maintained GUI base and candidate SHA before merge. Keep full #27 open and record only this exception's completion. Preserve #2, the consolidation sequence, #5/#10/#11/#38 guarantees, source and collection identities, ranking and the embedding contract. Do not inspect the real corpus, alter credentials, change historical refs, initiate migration, change Foreman configuration, or duplicate/load/download model weights.

Use the pinned audit guard and launcher from `docs/Research/issue-027` on `PYTHONPATH` against the maintained tree. Run from its root.

```powershell
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests/test_structural_answers.py
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests/test_rag_prompting.py tests/test_prompt_injection_guards.py tests/test_generation_cli.py tests/test_generation_qt.py tests/test_generation_streamlit.py tests/test_generation_models.py tests/test_streamlit_citations_config.py
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests/test_gui_direct_search_bm25.py tests/test_chroma_hybrid_retrieval.py tests/test_embedding_contract.py tests/test_chroma_embedding_entrypoints.py tests/test_streamlit_hybrid_embeddings.py tests/test_offline_search.py
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests/chroma_hybrid_native_cases.py
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests
py -3.12 -m py_compile rag_prompting.py generation_models.py GUI_workers.py GUI_chat.py rag_gui.py streamlit_app.py groq_lama_chromadb_RAG_ETTS.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```
