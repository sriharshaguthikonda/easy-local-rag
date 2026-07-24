# Issue #21: Thin CLI over the stable RAG service contract

[Roadmap ledger](README.md)

**Status:** Open; blocked pending stable retrieval/service contract.
**GitHub:** [#21](https://github.com/sriharshaguthikonda/easy-local-rag/issues/21)
**Parent epic:** #17.
**Labels / priority:** `priority:P1`, `type:enhancement`.

## Dependencies

Blocked by `.memory` #5 and #23: evidence search, hydration, fusion, structured
citations, and evaluation must pass their immutable handoffs first. Reciprocal
comments on [#21](https://github.com/sriharshaguthikonda/easy-local-rag/issues/21),
[#23](https://github.com/sriharshaguthikonda/easy-local-rag/issues/23), and
[`.memory` #5](https://github.com/sriharshaguthikonda/.memory/issues/5) must
record each provider's exact commit SHA, API/schema version, compatibility-test
command and passing result, and package name/version. No branch, `latest`,
placeholder, or mutable artifact satisfies this gate. It follows #20B and
precedes #27/#25/future GUI. Coordinate with #22 only at the provider
interface; #21 must not embed provider policy in retrieval.

## Implementation slices

1. **#21A port boundary:** own CLI/service ports, transport, request framing, and categorized errors; consume `SearchRequest`, `SearchHit`, `HydratedChunk`, `Citation`, `ChatRequest`, `ChatResponse`, and `ProviderStatus` from stable producers.
2. **CLI presentation:** implement explicit `status`, `search`, `show-chunk`, `neighbours`, `chat`, `ingest`, `migrate audit|export|verify`, and `doctor` commands; keep interactive shortcuts documented and turn-local.
3. **State/error discipline:** stable `--json`, categorized errors/non-zero exits, explicit/versioned session persistence, one stored user/assistant turn each, isolated evidence per turn, tokenizer-backed budget, and safe cancellation.
4. **Optional capabilities:** adapters for provider/TTS/STT stay optional; search-only and `--no-model` work without importing GUI, speech, or provider-specific modules.

`#21B` is the post-#23 integration slice. It may wire the verified CLI/service
port to retrieval and provider consumers, but cannot redefine `.memory` #5
retrieval/hydration, #22 provider behavior, or #27 answer validation.

## Affected interfaces, files, and artifacts

- Existing entry path `localrag.py`; future maintained `easy-rag` command and thin CLI module(s).
- Shared service contracts from `.memory` #5, including search, hydration, citation validation, and provider status.
- Stable JSON schemas for commands; doctor output with resolved non-secret configuration and privacy/provider status.
- Tests for no-model search, single-turn persistence, provider/database distinction, JSON compatibility, cancellation, and missing optional dependencies.

## Concrete actions

- Remove client-owned storage initialization/retrieval/context/provider/speech coupling from the maintained CLI path; clients never access Chroma/PostgreSQL directly.
- Make `search` evidence-only; `chat` invokes a selected provider only after retrieval/context construction; a model failure is an error, never a fabricated answer.
- Apply config precedence: CLI flags, project/user config, environment machine/secrets, safe defaults. `doctor` prints only resolved non-secret settings.
- Fence retrieved evidence by turn and do not carry prior chunks into future prompts automatically. Validate imported sessions before deserialization.

## Verification

```powershell
python -m pytest tests -q
python -m pytest tests -q -k "cli or search_only or session or json"
easy-rag status
easy-rag search "query" --json
easy-rag doctor
```

Use a fake service/provider to prove search does not call a model, failure leaves session state intact, JSON schema is stable, and absent TTS/STT packages do not prevent search.

## Closure gate

The immutable reciprocal dependency handoffs are recorded; the CLI searches
the PostgreSQL evidence store through those exact shared-service package/schema
versions without GUI/TTS/STT/provider-specific imports; retrieval is
fake-service testable; provider and database failures have distinct
exits/messages; existing command-line workflow remains usable during migration.

## Rollback constraints

Keep the old CLI available only as an explicitly marked legacy path during transition; do not route it to new storage implicitly. A failed client rollout returns users to the prior supported command while preserving sessions and not mutating evidence. Chroma rollback remains governed by #25.

## Commit boundary

One commit for service-contract adapters and CLI command behavior with focused tests; separate commits for session migration and optional speech/streaming integration.
