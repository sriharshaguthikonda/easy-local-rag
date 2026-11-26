# Issue #10: Search BM25 over the full corpus

[Roadmap ledger](README.md) · [Implementation packet](../superpowers/plans/2026-11-27-issue-010-chroma-hybrid-retrieval-implementation.md)

**Status:** OPEN — one-time maintained ChromaDB exception approved. Separate activation pending.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/10
**Labels / priority:** `priority:P1`, `type:perf`
**Dependencies:** [Owner approval](https://github.com/sriharshaguthikonda/easy-local-rag/issues/10#issuecomment-7200914319). The maintained GUI base includes independently verified #11. #2 remains open with its existing gate. PostgreSQL migration keeps its existing sequence.

## Acceptance

- Keep Chroma collection and chunk IDs through candidate retrieval, fusion, UI, and citations. Identical filenames never discard distinct chunks. Sort missing distances last in both directions and apply active filters before limits.
- Search the full eligible collection independently with BM25 and Chroma vector retrieval. Fuse deterministically, retain lexical-only hits and preserve explicit lexical-only search without embedding or generation.
- Verify addition, deletion, same-count edits, metadata filters, collection/store switch and same-name recreation. Keep memory bounded and prove consistent reads with a separate synthetic writer. Preserve the #11 embedding-model checks and read-only retrieval interface.
- The packet and separate activation PR merge before edits. Verify behavioral RED on the frozen base, focused and full tests including #11, and independent review and verification on the exact head. Close only after the verified implementation merges. Update the canonical ledger with evidence.

Implementation and verification use synthetic Chroma data. Roll back by reverting the isolated #10 code merge, preserving Chroma evidence IDs and the #11 guard.
