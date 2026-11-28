# Maintained request admission

PyQt, Streamlit, the maintained CLI entry points and their query rewrites use the same admission check. For each permitted model candidate, the router budgets the rendered text request and reserves the full requested output before generation. Context capacity and provider rate limits are independent. A capacity refusal keeps the evidence available in the structural answer paths.

## Verify a deployment profile

The shipped profile file carries routing policy and published limits. It ships without verified deployment assets. Generation stays unavailable until an operator supplies a matching profile. Search and retrieval remain available.

Copy [the supplied profiles](../config/generation-models.json) to a local file, keep the existing version and provider sections, and set EASY_RAG_GENERATION_PROFILES to its absolute path. Add an admission object to the matching model row. Record the evidence that links the files to the exact deployment before setting these fields.

| Field | Required value |
| --- | --- |
| provider, model, revision | Exact configured provider and model IDs plus a verified deployment revision. A pinned local manifest can supply the revision for Ollama. |
| tokenizer_path, tokenizer_sha256 | Absolute local path to a UTF-8 tokenizer.json and the lowercase SHA-256 of its bytes. Use an asset with verified parity for the deployed tokenizer, including special tokens and normalization. |
| template_sha256 | Lowercase SHA-256 of the deployed chat template. Groq uses the pinned Jinja file. Ollama uses the UTF-8 template returned by the local show endpoint. |
| add_special_tokens | Boolean matching the tokenizer invocation after rendering the deployment template. |
| max_added_tokens | Zero if the complete count and framing match the deployment. Otherwise enter a verified nonnegative bound for additional tokens the provider adds to each request. |
| template_path and template_variables | Groq only. Supply the absolute local path to the deployment's Jinja template. Optional template variables must match the deployment's defaults. The renderer receives the prepared messages, assistant generation marker and effective reasoning options. |
| server_version, model_digest | Ollama only. Use the verified native adapter version 0.34.0 and the SHA-256 digest of the installed model manifest. The process list and installed model must agree. |

Verify the model and revision with the deployment owner or another source that establishes parity. A similar model name, public tokenizer file or SHA-256 by itself does not establish which tokenizer and template a deployment uses. Check special tokens, assistant prefill, hidden framing and all template branches used by the requests. If any of those differ or the additional token count is unbounded, leave the profile unenrolled. The runtime does not download assets. Hosted public API metadata may omit revision and template identity. Enroll those routes only when you can independently establish parity. A client may fail closed again when a deployment changes.

## Effective limits and outcomes

- **Groq.** The model catalogue and published limits bound capacity. A profile or request can lower that ceiling. The full output reserve is sent as max_completion_tokens. Organization tokens per minute govern a separate upstream rate limit. [Model limits](https://console.groq.com/docs/models) · [API reference](https://console.groq.com/docs/api-reference) · [Rate limits](https://console.groq.com/docs/rate-limits).
- **Ollama.** The loopback adapter checks the installed and already-loaded model, model context metadata and loaded runner allocation. It requests a render-only preview with truncation and context shifting disabled and rechecks metadata before generation. Lower operator limits narrow admission while retaining the verified runner allocation in the request. Use the verified server version and a running model with available context metadata. [Context settings](https://docs.ollama.com/context-length) · [Pinned server implementation](https://github.com/ollama/ollama/blob/v0.34.0/server/routes.go#L2529-L2554).
- **Requests and fallback.** Only complete older user/assistant turns can be removed. System instructions, the current question and serialized evidence stay intact. Once selected for the first dispatch, the prepared messages stay identical across eligible same-provider attempts. Each replacement uses its own count, limit and output reserve.
- **Receipts.** Generation results and capacity errors carry the effective limit, reserve, count quality and provenance. An exact count reflects the verified renderer and tokenizer. A conservative count adds the verified upper bound for framing. Unknown counts refuse generation. Reported truncation, comparable usage discrepancies, capacity failures and incomplete output prevent the answer from being published. The structural validator then checks the answer against the prepared evidence. Client hashes, token usage and a normal provider stop establish no proof that the provider consumed the entire request.
