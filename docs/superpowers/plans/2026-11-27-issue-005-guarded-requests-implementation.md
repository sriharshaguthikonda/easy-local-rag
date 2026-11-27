# Issue #5 guarded requests implementation packet

**State** Locked for the packet PR. The separate activation PR authorizes execution.
**Mode** Maintained ChromaDB.
**Target** `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
**Frozen base** `89514bf502a39776b2261ff1657146f530364765`.
**Authority** [Issue #5 approval](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5#issuecomment-7213652317) · [Canonical acceptance](../../issues/ISSUE-005-prompt-injection.md) · [Ledger](../../issues/README.md).

## Locked boundary

- `rag_prompting.py` owns the request. `build_guarded_request(system_message, question, sources, history=(), *, task_instruction="", citation_instruction="")` returns an ordered list of role/content dictionaries. The first is the application system message with the evidence rules and any code-owned mode and citation instructions. Carry forward text-only user and assistant history, preserving order. Construct the final user message once from ordered evidence and the original question. Keep the input dictionaries and strings intact.
- Number raw sources once in retrieval order using the existing numbering helper. Require a non-empty document and unique positive integer citation IDs before generation. Source dictionaries preserve the exact original text and filename metadata and any supplied collection and chunk IDs. Qt normalizes its nested Chroma metadata to this shape. Keep the stable ID pair in the application's citation/source records.
- Keep the `retrieved_context` and `source` containers. The application controls all markup and the numeric source ID attribute. Encode each source as a JSON object with `file_name` and `document` inside its wrapper. Encode wrapper-significant `&`, `<` and `>` as JSON Unicode escapes after standard JSON encoding. Use the same encoding for the question as a JSON string outside the evidence container. Preserve Unicode, quotes, backslashes, newlines and role-like text on decode. Exact source citation numbering remains request-local and ordered. Raw text and metadata in storage are unchanged.
- `GUI_workers.ChatWorker`, Streamlit `_prepare_generation`, and `_generate_chat_result` in both CLIs call the shared composer. Keep Streamlit's original ordered source contract, modes, citation acceptance, and current-message budget behavior across synchronous and worker-backed generation. Both CLIs use the ordered source records rather than their joined projection. Missing or malformed RAG evidence fails before generation with the existing safe error surfaces. Search-only paths bypass generation and speech.
- History append occurs only after an accepted completion. Persist the original user question and assistant text. Capabilities stay in trusted application configuration. Disallow tool options and reject provider-declared tool output in guarded answers with a typed, terminal generation error. Preserve generic router behavior outside the text-only RAG request, caller cancellation, and existing same-provider pre-output model fallback. Retry receives the same guarded message sequence and tool policy.
- Cloud speech starts off. Qt and Streamlit require a labeled session opt-in, and each CLI requires the explicit `--allow-cloud-speech` flag or an explicit keyword opt-in from a trusted programmatic caller. Provider choice and playback suppression signals cannot grant consent. Route model and source text to plain text or escaped text in maintained GUI display paths, including streaming output and restored history. Explicit file opening and copying remain user actions.
- Only code-owned policy enters the system role. Prior assistant text can carry document-derived instructions and remains assistant text on replay. The controls preserve structural roles and application permissions. The model can still interpret instructions within source or history strings.
- #27 semantic claim validation, the full #22 provider abstraction, #25 retirement, PostgreSQL migration and MCP integration have separate scope.

## Owners and RED acceptance

- **A** owns `rag_prompting.py`, `tests/test_rag_prompting.py`, and replacement of source-text guard checks with behavioral checks in `tests/test_prompt_injection_guards.py`. Use inert markers for equivalent delimiter-shaped text, quote and ampersand metadata, Unicode, line endings, role-shaped text, and secret/tool-shaped requests. Assert decoded identity, exact wrapper counts, order, immutable source values, and system versus user role. Exercise existing functions on the frozen tree so failures expose the defect.
- **B** owns both maintained CLI scripts and `tests/test_generation_cli.py`. Capture both provider choices through both CLI entry points and the programmatic answer wrappers. Assert raw text stops at the boundary, the user turn contains current evidence exactly once, follow-up traffic excludes stale evidence, and the no-consent path never dispatches speech. Keep the no-corpus offline control and the Groq rewrite/Ollama embedding separation.
- **C** owns `GUI_workers.py`, `GUI_chat.py`, `rag_gui.py`, `streamlit_app.py`, `tests/test_generation_qt.py`, `tests/test_generation_streamlit.py`, and affected citation request assertions. Capture Qt and Streamlit requests for both providers, all Streamlit modes, and both Streamlit request paths. Assert trusted role placement, fresh source maps, accepted turns and source IDs. Spy on display and speech without loading external resources. Preserve first-party UI interactions and offline results.
- **D** owns the RAG text-only option and output handling in `generation_models.py` and `tests/test_generation_models.py`. Use fake streaming and non-streaming responses with text or tool-shaped events. Record RED for an accepted tool-only completion where present. Assert zero dispatch, text-only rejection, no tool-authorized side effect, and identical message/policy on permitted fallback. Exercise both provider adapters and confirm generic tool-capable calls retain their existing shape.
- **E** independently reviews and verifies the exact implementation candidate. Review both role/content and authority boundaries, stale evidence, state and fallback, rendering and opt-in egress, #10/#11 invariants and failure paths. Codex verifies every material claim and any correction at the new head.

Exercise literal Unicode escape sequences and forged citation IDs in round-trip tests. Preserve Streamlit's single-evidence-source contract by mapping every accepted citation to exactly one ordered source record. Verify the current evidence and system message survive budget trimming intact, or stop before dispatch. On fallback, assert the endpoint, account, locality, options and prepared roles remain fixed and the router rechecks capacity. Reject mixed text and tool responses before committing an answer or dispatching speech. Cover cancelled and partial streams. Verify speech opt-in names the destination and disabling it prevents further starts from pending Qt playback. Cover the restored and streamed text renderers with inert Markdown or HTML fixtures and test the local no-egress default.

## Commands and gates

After the activation PR merges, run the focused command on the frozen checkout with only the test changes present. Expect specific assertion failures for the defects named above and passing controls where the old app already blocks an action. Record the failing nodes and expected-versus-actual effects. Commit the RED tests separately before editing product code.

Run the same focused command after the fix and require every case to pass. Then run the full and native suite and compilation under a Python 3.12 socket-denial guard inherited by child Python processes, with provider credential variables unset in the test process, Qt offscreen, Chroma telemetry disabled and ChromaDB 1.0.16. Use pytest temporary Chroma stores only.

```powershell
py -3.12 -m pytest -p no:cacheprovider tests/test_rag_prompting.py tests/test_prompt_injection_guards.py tests/test_generation_cli.py tests/test_generation_qt.py tests/test_generation_streamlit.py tests/test_generation_models.py tests/test_streamlit_citations_config.py -q
py -3.12 -m pytest -p no:cacheprovider tests/test_gui_direct_search_bm25.py tests/test_chroma_hybrid_retrieval.py tests/test_embedding_contract.py tests/test_chroma_embedding_entrypoints.py tests/test_streamlit_hybrid_embeddings.py tests/test_offline_search.py -q
py -3.12 -m pytest -p no:cacheprovider tests/chroma_hybrid_native_cases.py -q
py -3.12 -m pytest -p no:cacheprovider tests -q
py -3.12 -m py_compile rag_prompting.py generation_models.py GUI_workers.py GUI_chat.py rag_gui.py streamlit_app.py groq_lama_chromadb_RAG_ETTS.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

The model smoke uses a locally installed, manifest-verified Ollama model on an already available local endpoint and synthetic source records in memory. Disable speech and provide no tools or credentials. The fixture has a harmless evidence-backed answer and an unrelated directive to output a different inert marker. Record the selected model, exact candidate, presence of the expected evidence answer and absence of directive compliance. The result is specific to that model and fixture. Leave the smoke gate pending if a permitted model is unavailable. Hosted model smoke requires the existing provider and credential gates and is outside this packet's execution.

Docs review and merge, separate activation merge, tests-only RED commit, implementation and any review fixes, independent exact-head verification, GUI merge, evidence and closure, and ledger reconciliation are distinct boundaries. Keep #2 open throughout. Do not publish test credentials, corpus data, or private traces in receipts.

## Rollback

Use the canonical guarded rollback constraint. Disable optional display or speech behavior if necessary while preserving the shared guarded request and the Chroma retrieval contract.
