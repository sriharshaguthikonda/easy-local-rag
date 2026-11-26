# Issue #10: Search BM25 over the full corpus

[Roadmap ledger](README.md) · [Implementation packet](../superpowers/plans/2026-11-27-issue-010-chroma-hybrid-retrieval-implementation.md)

**Status:** CLOSED — independently verified and merged in PR #62.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/10
**Labels / priority:** `priority:P1`, `type:perf`
**Dependencies:** [Owner approval](https://github.com/sriharshaguthikonda/easy-local-rag/issues/10#issuecomment-7200914319). The maintained GUI base includes independently verified #11. #2 remains open with its existing gate. PostgreSQL migration keeps its existing sequence.

## Acceptance

- Keep Chroma collection and chunk IDs through candidate retrieval, fusion, UI, and citations. Identical filenames never discard distinct chunks. Sort missing distances last in both directions and apply active filters before limits.
- Search the full eligible collection independently with BM25 and Chroma vector retrieval. Fuse deterministically, retain lexical-only hits and preserve explicit lexical-only search without embedding or generation.
- Verify addition, deletion, same-count edits, metadata filters, collection/store switch and same-name recreation. Keep memory bounded and prove consistent reads with a separate synthetic writer. Preserve the #11 embedding-model checks and read-only retrieval interface.
- The packet and separate activation PR merge before edits. Verify behavioral RED on the frozen base, focused and full tests including #11, and independent review and verification on the exact head. Close only after the verified implementation merges. Update the canonical ledger with evidence.

Implementation and verification use synthetic Chroma data. Roll back by reverting the isolated #10 code merge, preserving Chroma evidence IDs and the #11 guard.

## Verification

The [packet PR #60](https://github.com/sriharshaguthikonda/easy-local-rag/pull/60) and
[activation PR #61](https://github.com/sriharshaguthikonda/easy-local-rag/pull/61) merged before product edits.
The tests-only commit `393309b` reproduced nine failures. The exact reviewed
head `55a915433e44f5cf0be66453d57a4e9670784021` and GUI merge
`89514bf502a39776b2261ff1657146f530364765` have the same tree. [Implementation PR #62](https://github.com/sriharshaguthikonda/easy-local-rag/pull/62)

In a clean checkout with provider access denied, 344 full-suite tests passed.
The focused #10 repeat passed 13 tests including 11 native Chroma scenarios
run in isolation. The #11 suite passed 90 tests. A separate process verified
that a reader waits for a managed writer and sees a coherent same-count edit.
The native watcher probe verified one guarded write and preserved evidence IDs
and source path. [Closure evidence](https://github.com/sriharshaguthikonda/easy-local-rag/issues/10#issuecomment-7201734064)

Filtered vector retrieval ranks eligible stored embeddings using the configured
distance metric after the synthetic fixture exposed omissions in Chroma 1.0.16
filtered ANN. Unfiltered vector search uses Chroma ANN. The packet records the
implementation decision and the bounds measured on synthetic data.
