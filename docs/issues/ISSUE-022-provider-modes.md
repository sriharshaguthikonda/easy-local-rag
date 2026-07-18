# Issue #22: Explicit provider modes and local/cloud truthfulness

**Status:** Open; follows the thin service boundary.
**GitHub:** [#22](https://github.com/sriharshaguthikonda/easy-local-rag/issues/22)
**Parent epic:** #17.
**Labels / priority:** `priority:P1`, `type:security`.

## Dependencies

Requires #2 secret remediation, #20B's destination integration, and the stable service contract from `.memory` #5/#23. Provider-core work may proceed in parallel with #21; `status`, `doctor`, and `chat` integration follows #21's client surface. It runs before #27 grounded-answer validation and #25 cutover. It protects offline/local operation throughout and must not delay evidence-only search.

## Implementation slices

1. **Provider contracts:** separate embedding, generation, reranker, STT, and TTS adapters; expose name, endpoint, local/remote classification, model/version, capabilities, health, timeout/retry, logging, limits, and tokenizer support.
2. **Explicit modes:** `offline-search` has no network/model/speech/telemetry; `local-chat` uses configured local endpoints and fails closed; `cloud-chat` requires selected provider plus disclosure/preview/redaction; `mixed-speech` lists remote speech/TTS and is never described as fully local.
3. **Egress and secrets:** use environment/OS secret store, non-secret sample values only in `.env.example`, startup missing-name checks, redacted diagnostics, all-ref scan/rotation under #2, and per-request egress summary.
4. **Failure behavior:** no implicit fallback between provider classes; distinct retrieval/provider errors; provider loss leaves search-only usable; real provider/tokenizer limits drive budgets/rate controls.

## Affected interfaces, files, and artifacts

- Provider adapter interfaces and provider status consumed by #21 `status`/`doctor` and `chat`.
- Mode configuration schema, safe `.env.example`, non-secret resolved config/egress display, and README mode wording.
- Egress record: selected provider, query inclusion, chunk count, source-name redaction state, approximate tokens, and history inclusion.
- Tests: network-free offline mode, no cloud fallback in local mode, approved-cloud-context boundary, redacted secrets, optional dependency absence, limits/tokenizer use, timeout/cancellation.

## Concrete actions

- Require an explicit requested mode and provider selection before generation/speech calls; never infer cloud fallback because a local endpoint failed.
- Keep retrieval independent: every mode can run search-only with no model/speech dependencies.
- Before cloud dispatch, present or safely log the egress record and apply configured source/metadata omission/redaction.
- Redact secret values from exceptions, exports, diagnostics, screenshots, example commands, and prompts. Report missing variable names only.
- Make all mode claims truthful: Groq, Google recognition, Edge TTS, or any remote service makes the relevant path cloud-assisted/mixed.

## Verification

```powershell
python -m pytest tests -q
python -m pytest tests -q -k "provider or offline or egress or secret"
easy-rag status
easy-rag doctor
```

Run an offline-mode network-denial test; simulate a dead local endpoint and assert no cloud call; capture a cloud request through a fake adapter and assert only approved/redacted context is sent; inspect diagnostics for redaction.

## Closure gate

`status`/`doctor` truthfully identify offline, local, cloud-assisted, or mixed operation; offline search makes no network calls; local failures never fall back to cloud; cloud egress is explicit/testable; no secrets appear in tracked files or diagnostic output; and missing optional providers do not break CLI search.

## Rollback constraints

Mode changes must be reversible configuration changes and must preserve a safe `offline-search` path. Never auto-retry a failed local call through cloud, and never retain sensitive prompt/egress logs by default. #2 controls credential rotation; #25 controls data-platform rollback.

## Commit boundary

Commit provider contracts/mode selection and their fake-adapter tests together; separate commits for each optional provider integration and for README/config documentation.
