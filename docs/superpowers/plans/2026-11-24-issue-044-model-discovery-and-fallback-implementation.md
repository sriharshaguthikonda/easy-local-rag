# Model discovery and same-class fallback Implementation Plan

> **For agentic workers:** Run the tasks through test-driven development after the separate activation merge.

**Goal:** Recover the maintained RAG paths from stale generation model IDs and report the model that answered.
**Architecture:** Add a shared catalogue and bounded retry policy at the existing Groq/Ollama SDK calls. Keep transport, retrieval and presentation with their existing owners.
**Tech Stack:** Python, Groq SDK, Ollama SDK, PyQt5, Streamlit, pytest.
**Spec:** [ISSUE-044](../../issues/ISSUE-044-model-discovery-and-fallback.md) · [Ledger](../../issues/README.md) · [Roadmap](../../ROADMAP.md) · [Live issue](https://github.com/sriharshaguthikonda/easy-local-rag/issues/44).

## Branch and safety

The target is `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
The frozen planning base is `f78d53476a452dd66aa207d74ebd7d2dca458fca`, the
published #38 closeout tip. Freeze the separately merged activation commit
as the implementation base and record its full SHA with review evidence.
The GUI lineage is authoritative. Keep other local branches and owner files outside these PRs. Keep #2 and the #18/#26B/migration gates.
Run the ledger validator before activation and closure and after disposition
changes. Owner approved this independent exception on 2026-11-24.

## Existing implementation to reuse

Use the normalized catalogue, freshness check, unverified-candidate
exclusion, eligibility-before-ranking and safe attempt-status patterns from
Model Bridge at `56519023e22f6f7a91f36a7074352c771bdc4176`,
`src/model_bridge_mcp/discovery.py` and `src/model_bridge_mcp/router.py`.
Verified its `tests/test_discovery.py` and `tests/test_router.py` locally
with `python -m pytest -p no:cacheprovider tests/test_discovery.py tests/test_router.py -q`
and `PYTHONPATH=src` (23 passed). Adapt the design to the existing SDKs and
preserve the app's provider choice. Model Bridge's across-provider routing,
on-disk cache, name-based guesses, and broad failure classification are
unsuitable here. [Alternative implementations](../../Research/landscape.yaml).

## Caller map

| Caller | Ownership |
|---|---|
| `GUI_models.py`, `GUI_settings.py`, `rag_gui.py` | Populated menus, scoped refresh, preference, provider, strict pin and cooperative cancellation. Ollama and Groq have separate menus already. |
| `GUI_workers.py`, `GUI_chat.py` | Stream and model status, cancel, single history append, evidence retained on failure. Replace the broad Groq-to-Ollama catch. |
| `GUI_context.py`, `GUI_direct_search.py` | JSON-mode Groq rewrites and lexical-only offline search. Reuse existing lexical scoring and query text when rewrite is unavailable. |
| `streamlit_app.py` | Explicit provider, discovered preference, refresh, strict pin, evidence and model status. Replace the broad catch-and-switch. Keep citation and return-shape contracts and expose available evidence on generation failure. |
| `groq_lama_chromadb_RAG_ETTS.py`, `streamlit_groq_lama_chromadb_RAG_ETTS.py` | Streaming answer and JSON rewrites, existing provider args plus list/refresh/model/pin flags. Keep CLI cancellation and suppress startup generation probes. |
| `context_retrieval.py` | Route its live legacy Groq rewrite through the shared fallback boundary and reuse raw query on failure. Preserve the existing embedding contract. |
| Ingestion and TTS | Continue to use their existing embedding and speech contracts. |

## Shared boundary

- Add `generation_models.py` and `config/generation-models.json` (schema
  version 1). Normalize catalogue records to ID, provider, task class(es),
  operating mode, quality band, context/output limits, price tier and input/
  output ceiling, ranking and capability flags. Separate text chat, explicit
  reasoning, embedding and speech. Distinguish text, JSON-object and strict
  JSON-schema requests. Apply the more restrictive of provider metadata and the
  profile whenever both report a limit. List unknown IDs as unprofiled and
  keep them out of automatic selection.
- A long-lived catalogue per provider client/session uses in-memory monotonic
  TTL of 300 seconds and serializes its refresh. Scope it to provider,
  endpoint and authenticated client/account. Keep keys and raw errors out of
  cache entries and diagnostics. Distinguish empty, error and unverified saved
  IDs. Manual refresh bypasses TTL. Discovery timeout is 10 seconds.
- Use the authenticated Groq SDK models list and active flag. Use Ollama list
  and model show for installed models, completion and other capabilities,
  and context metadata. Local routing permits local endpoints/models only.
  Reject remote/cloud manifest metadata including cloud aliases before
  dispatch. Bulk model metadata follows the catalogue TTL. Recheck the
  selected Ollama manifest before each dispatch, since a local tag can
  change during that TTL. Use raw response metadata to preserve remote
  fields dropped by older Ollama SDK response types.
- Require app provider/mode, task class, quality and budget policy plus
  requested capabilities before deterministic rank by configured priority
  then ID. Missing policy fails closed. Plain answer and explicit reasoning
  are distinct request classes. Groq's recommended GPT-OSS 120B serves
  general text chat only through an approved answer-only profile with hidden
  reasoning, using the original service tier and a verified cost ceiling.
  If the request requires reasoning disabled or the ceiling cannot be proven,
  exclude that profile. Keep the requested tier on the wire and reject
  opportunistic service-tier changes.
  [Groq pricing](https://console.groq.com/docs/models) ·
  [Groq reasoning](https://console.groq.com/docs/reasoning).
- Attempt the preferred eligible ID first, with max 3 generation calls per
  request and SDK automatic retry disabled on that boundary. Try another
  candidate only for a reliable typed unavailable-model error. Groq handles
  status+structured error code and Ollama handles its typed error only with
  model-specific missing-model evidence. Exclude failed IDs for the request
  even when listed again. Never retry after application-visible text, tool
  deltas, TTS or a completed side effect. Treat other errors as terminal and
  distinguish discovery failure and no candidate. Close abandoned streams.
  Check cancellation before discovery, before each attempt and between
  chunks. Set bounded transport timeouts.
- Build the current request once. Keep the approved data egress and evidence
  text/IDs invariant across candidates. Validate the whole prepared input
  and reserved output per candidate before calling it. Reuse token counting
  with a conservative UTF-8 byte and wire-overhead bound where tokenizer or
  chat template is not reliable. Apply the local configured effective context
  as well as the model limit. A candidate requiring evidence or history
  truncation is ineligible. Validate citations before committing the final
  answer/history and avoid duplicate chunks and TTS across attempts.
- Use catalogued ID plus safe category for provider diagnostics. Keep the
  saved preference on failure and display actual model/reason separately.
  Search-only lexical mode opens local persisted documents with telemetry
  disabled and uses the original query, no Groq rewrite, embeddings, speech
  or provider clients. Existing vector search continues to assert its
  embedding contract when explicitly used.

## Call interface

`generation_models.groq_router(client=None)` and
`generation_models.ollama_router(client=None, host=None)` accept existing SDK
clients for tests. Production callers keep a router for the life of an
account/session. The factories defer SDK creation so offline operations never
require a key. `router.catalogue(refresh=False)` returns an ordered list of
generation profiles with model ID and capabilities. A safe
`GenerationError` distinguishes catalogue failure from successful emptiness.

`router.generate(preferred=..., messages=..., strict=False, stream=True,
max_output_tokens=1024, cancelled=..., on_chunk=..., **provider_options)`
returns a `GenerationResult` with `text`, `preferred_model`, `actual_model`,
and `reason`. Infer capability requirements from the requested options,
including tools, response format and streaming. Token ceilings apply to
every dispatch. Pass temperatures and other supported provider options
through the same SDK, never through the other provider. Report a safe error
category on failures and keep retrieval/evidence with the calling UI.

## Profile schema and first supported models

The config is `{"version": 1, "providers": {"groq": {MODEL_ID: PROFILE},
"ollama": {MODEL_ID: PROFILE}}}`. The user can override this file via
`EASY_RAG_GENERATION_PROFILES`. A profile has `classes` (a list of
`chat` and/or `reasoning`), `quality` (a string), `tier` (Groq
`on_demand` or Ollama `local`), nonnegative `input_usd_per_mtok`
and `output_usd_per_mtok`, positive `context_tokens` and
`output_tokens`, integer `priority` (lower first), and literal
capabilities from `stream`, `tools`, `json_object`, `json_schema` and
`reasoning_off`. Embedding and speech records are excluded. Invalid
or unknown profiles cannot be used for fallback. The replacement must
match class, quality and tier. Each USD limit must be at most the
preferred profile's limit. Apply the same maximum output count on every
attempt, including reasoning tokens when the provider uses them.

| Groq ID | Class and quality | On-demand input/output USD per million | Context/output tokens | Priority |
|---|---|---|---|---|
| `llama-3.3-70b-versatile` | chat, large | 0.59 / 0.79 (legacy ceiling) | 131072 / 32768 | 10 |
| `openai/gpt-oss-120b` | chat and reasoning, large | 0.15 / 0.60 | 131072 / 65536 | 20 |
| `llama-3.1-8b-instant` | chat, compact | 0.05 / 0.08 (legacy ceiling) | 131072 / 131072 | 10 |
| `openai/gpt-oss-20b` | chat and reasoning, compact | 0.075 / 0.30 | 131072 / 65536 | 20 |

Use Groq's verified standard on-demand tier explicitly for fallback.
Enterprise or opaque negotiated pricing requires an explicit profile.
The Llama rows are legacy standard-spend ceilings. Groq's current
model pages list enterprise access separately. GPT-OSS 120B may serve
plain chat with `include_reasoning=False` and low reasoning effort.
It lacks the `reasoning_off` capability. No tools run implicitly. GPT-OSS
20B exceeds the old compact model's ceilings, so an unavailable default
rewrite continues with the original query, unless the user explicitly
selects another rewrite model. [Groq service tiers](https://console.groq.com/docs/service-tiers) ·
[Groq model details](https://console.groq.com/docs/models) ·
[Groq structured output](https://console.groq.com/docs/structured-outputs).

For Ollama use server metadata for task class and effective context, capped
at the request's explicit `options.num_ctx` (default 4096), and profile
quality/priority. Include reviewed zero-cost local compact profiles
`phi-3`, `phi3:latest`, and `llama3.2:3b`, with priorities 10, 20 and 30,
a 4096-token context limit and a 1024-token output ceiling. Local IDs
reported as installed completion models can appear for explicit selection
when unprofiled. Require a configured profile before an automatic fallback.
Inspect `remote_model` and `remote_host` before dispatch. Check Ollama
server metadata for all candidates and cap local output through
`options.num_predict`. [Ollama schema](https://github.com/ollama/ollama/blob/main/docs/openapi.yaml).

## Implementation tasks and seams

1. **Router and policy.** Own `generation_models.py`,
   `config/generation-models.json` and `tests/test_generation_models.py`.
   Use fake client catalogue/stream objects and injected monotonic clock.
   Assert the ordered attempts `[llama-3.3-70b-versatile, openai/gpt-oss-120b]`
   after a real SDK-shaped 404 body. Assert event order is initial call,
   forced catalogue refresh through the same client, then fallback call.
   Have the initial catalogue omit GPT-OSS and the refresh add it, and
   assert exact message text/citations, `on_demand` on both calls, and
   result status naming the replacement.
   Test two simultaneous account clients, forced refresh and TTL, and assert
   no retry after auth failure, other 404, cancellation or the first
   nonempty delta. Deny candidates on class, quality, remote manifest,
   paid limit, context, strict JSON, tools, or max output. Assert the call
   count stays at three after a sequence of typed pre-output failures.
2. **Offline lexical search.** Own `offline_search.py` and its tests.
   Export `search_documents(collection, query, limit)` returning existing
   source dicts with `id`, `document`, `metadata`, and
   `relevance_score`. Use the existing selected persistent collection's
   `get(include=["documents", "metadatas"])` and reuse BM25Okapi scoring
   from the direct search path on raw query text. Corpus is local and
   telemetry is disabled on the reader. For the network-denial fixture,
   use temporary Chroma data with document IDs `doc-beta` and
   `doc-alpha` and a unique term in `doc-alpha`. Assert that ID and exact
   document on search, reject every attempted socket, and assert no model
   client factory or embedding function is called.
3. **Qt.** Own `GUI_models.py`, `GUI_settings.py`, `rag_gui.py`,
   `GUI_chat.py`, `GUI_workers.py`, `GUI_context.py`, `GUI_direct_search.py`
   and `tests/test_generation_qt.py`. Persist `generation_provider`
   (`groq` or `ollama`, default `groq`) and
   `strict_generation_model` (bool, default false) in existing settings.
   The provider combo scopes active model discovery. The model menu keeps
   unavailable preferences visible and displays a distinct availability
   message. Use Qt interruption instead of terminating the worker. The
   existing Search Only button takes the lexical path, never instantiates
   a provider, and still works after failed answers. Local generation uses
   the original query for retrieval instead of a Groq rewrite. Exercise the
   model mixin and worker with injected fake routers and Qt offscreen,
   including the selected preference after account refresh and the status
   emitted by a successful fallback.
4. **Streamlit.** Own `streamlit_app.py`,
   `tests/test_generation_streamlit.py` and the existing Streamlit
   citation/config tests that use the affected callable. Store
   `generation_provider`, `strict_generation_model` and
   `generation_model_groq` / `generation_model_ollama` in session state.
   Feed the current selector and a scoped router to `chat_with_model`.
   Declare search-only in the sidebar and route it to lexical search.
   Keep a distinct last-model status and expose retrieved evidence on
   generation-only errors. Move provider creation to lazy session helpers
   and accept injectable router, Streamlit presentation and retrieval
   dependencies at the generation boundary. Exercise the existing
   `chat_with_model` callable, including #38 exact evidence and citation
   verification. Test workflow wiring with Streamlit test stubs and fake
   provider clients with outbound sockets denied.
5. **CLI and shared legacy backend.** Own
   `groq_lama_chromadb_RAG_ETTS.py`,
   `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `context_retrieval.py` and
   `tests/test_generation_cli.py`. Add `--provider {groq,ollama}` (default
   Groq), `--list-models`, `--refresh-models`, `--strict-model` and
   `--offline-search QUERY`. With explicit `--provider`, `--model`
   sets the selected provider's preference. Without `--provider`, keep
   the legacy Ollama-only meaning of `--model` and default to Groq with
   no cross-provider retry. Parse before
   opening a collection or constructing provider clients. Lazy-init the
   Groq and Ollama SDK clients after the corresponding provider is
   selected and export their pure generation functions through the
   shared boundary. Use test transport injection and the real CLI
   callable or a subprocess with synthetic dependencies. Assert explicit
   Ollama mode creates no Groq client, default mode creates no Ollama
   generation client, and the `context_retrieval.py` legacy rewrite uses
   the shared router under a fake 404. Keep the
   maintained-backend import restriction from AGENTS in test processes.
   Do not add source-string assertions. Assert preferred/actual/reason
   on the CLI and exact evidence across the fallback. Keep speech
   downstream of accepted output and honor Ctrl+C.

## Execution and tests

From the isolated checkout in PowerShell, set
`$ragPython = (Resolve-Path ../easy-local-rag/easyrag/Scripts/python.exe).Path`
and `$env:PYTHONDONTWRITEBYTECODE = '1'` to avoid touching tracked cache
files. This linked checkout has no venv. Use the existing repository
venv read-only and set the subprocess working directory to the linked
checkout. In tasks 1–5, run the focused test before production changes and
record the assertions that failed. Run it again for green.

```powershell
& $ragPython -m pytest tests/test_generation_models.py -q
& $ragPython -m pytest tests/test_offline_search.py tests/test_generation_qt.py -q
& $ragPython -m pytest tests/test_generation_streamlit.py tests/test_streamlit_citations_config.py -q
& $ragPython -m pytest tests/test_generation_cli.py -q
& $ragPython -m pytest tests -q
& $ragPython -m py_compile generation_models.py offline_search.py GUI_models.py GUI_settings.py rag_gui.py GUI_chat.py GUI_workers.py GUI_context.py GUI_direct_search.py streamlit_app.py groq_lama_chromadb_RAG_ETTS.py streamlit_groq_lama_chromadb_RAG_ETTS.py context_retrieval.py tests/test_generation_models.py tests/test_offline_search.py tests/test_generation_qt.py tests/test_generation_streamlit.py tests/test_generation_cli.py tests/test_streamlit_citations_config.py
& $ragPython scripts/validate_issue_ledger.py --repo sriharshaguthikonda/easy-local-rag
git diff --check "$activationSha..HEAD"
gitleaks git --redact=100 --no-banner --log-opts="$activationSha..HEAD" .
```

Resolve `$activationSha` to the exact merged activation SHA before
execution. No additional test failures are allowed. The frozen GUI tip
failed four mocks in `test_rag_gui_comprehensive.py` covering
`test_on_error_re_enables_buttons`,
`test_on_response_complete_re_enables_buttons`,
`test_on_response_complete_cleans_up_worker`, and
`test_on_search_results_populates_table`. It also failed the stale
`test_validate_issue_ledger.py` snapshot. The planning change refreshes
the snapshot for the #38 closure and #44 addition (20 ledger tests pass).

## Review, merge and rollback

Use a docs PR followed by a separate activation PR. From the recorded
activation SHA, commit behavioral tests and implementation with their
recorded RED/GREEN boundary. After exact-head code review and independent
verification, recheck the frozen base and use a GitHub merge commit. Close
#44 only after merge and ledger validation, with a closure-ledger update
and a #22 handoff naming the implemented contract and SHA. No live
credentialed smoke tests are in this packet. Keep rollback within #44 via
`git revert -m 1` of its implementation merge if required, or use strict
pin or lexical search through the new UI. No data conversion is required.
