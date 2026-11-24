# Issue #44 — Model discovery and same-class fallback

**Status:** Active — owner-approved independent P0 scope.
**GitHub:** [#44](https://github.com/sriharshaguthikonda/easy-local-rag/issues/44).
**Packet:** [Implementation and verification](../superpowers/plans/2026-11-24-issue-044-model-discovery-and-fallback-implementation.md).
**Ledger:** [Required execution order](README.md#required-execution-order).

## Failure and scope

Groq returns 404 `model_not_found` for the saved `llama-3.3-70b-versatile`.
Groq retired this ID on its free and developer tiers on August 16, 2026. It
publishes account-specific availability through its models endpoint.
[Groq changelog](https://console.groq.com/docs/changelog).

Fix the maintained Qt, Streamlit and CLI generation and retrieval-rewrite
calls in place. Chroma embedding, ingestion, and speech models are outside
the generation catalogue. The owner approved this as the sole new P0
implementation exception on November 24, 2026, after #37 and #38 closed.
The approval covers network-free verification. #2 remains open and its
credential, history remediation, and migration gates still apply.

## Acceptance contract

- Populate model menus from live provider catalogues on startup or provider
  selection. Keep the saved model selected if available, show missing saved IDs
  as such, and refresh without restart. A catalogue outage and an empty list
  must produce different messages.
- Keep the explicitly selected provider, operating mode, model task class and
  pricing tier. Bound spend by the approved input/output budget. Rank eligible
  models from versioned, configurable profiles plus provider metadata. Require
  context and output capacity, streaming, tools and structured-output
  capabilities as requested. Unknown profiles are ineligible for automatic use.
- Refresh after a typed model-unavailable error and retry at most twice with
  untried eligible models. Retain strict pin and cancellation. Reject account
  auth errors, unrelated 404s, partial-stream replay, provider or tier jumps,
  and embedding-model substitution.
- Carry the approved current request and exact evidence IDs/text through every
  retry. Recompute budget for the attempted model, abstain if the full request
  cannot fit, and preserve citation validation. Record selected/actual model,
  provider and any switch reason with safe diagnostics. Search-only must work
  when generation fails, and offline search must run without network calls.
- Cover the reported 404 and the no-fallback cases through network-free builds.
  Record the reuse source, test output, independent exact-head review,
  verification, merge SHA and rollback in the closure evidence. Validate the
  issue ledger immediately before activation or closure and after each state
  change.

[#22](ISSUE-022-provider-modes.md) consumes the verified routing/profile
contract during consolidation. #44 does not change #22's dependencies.
