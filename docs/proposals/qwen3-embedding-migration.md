# Proposal: Qwen3 Embedding 0.6B migration

**Status:** Superseded — GitHub issues #17 and #20 are the canonical tracking surface. This preserved proposal is historical planning context, not an implementation authority.
**Superseded by:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/17 and https://github.com/sriharshaguthikonda/easy-local-rag/issues/20

## Preserved proposal

Migrate dense ingest and query together to `qwen3-embedding:0.6b` in a **new** Chroma path and collection. Store the model, 1024-dimensional output, context length, chunk profile, and cosine space in collection metadata. Do not reuse an existing mxbai collection merely because vector dimensions match.

The initial migration creates a Qwen-named store, validates returned dimension before add, leaves the mxbai store untouched, then moves GUI/Streamlit query defaults only after ingest is proven. Full-corpus BM25 remains a separate sparse retrieval concern (issue #10); do not add sparse vectors to Chroma solely for this migration.

## Historical acceptance and rollback constraints

- Query model and collection metadata must match (issue #11).
- No vector truncation, projection, rounding, or cross-store mixing.
- Preserve the old collection; rollback is configuration selection, not deletion or relabeling.
- Validate model availability, one temporary-folder ingest, collection metadata/count, and GUI/Streamlit query before any broader reindex.

## Historical commit boundary

`docs: plan qwen3 embedding migration`; implementation commits follow only the canonical GitHub issue scope.
