# Issue #13 request admission for maintained ChromaDB

[Roadmap ledger](README.md) · [GitHub issue](https://github.com/sriharshaguthikonda/easy-local-rag/issues/13)

**Status:** Closed for the maintained ChromaDB scope. The owner approved a one-time maintained ChromaDB exception on 28 November 2026 while #2 remains open. The [pinned audit](../Research/issue-013-token-context-audit.md) and [canonical implementation packet](../superpowers/plans/2026-11-28-issue-013-request-admission-implementation.md) define the verification, target and activation gate.

All maintained request entry points use candidate-specific admission. The complete request and reserved output must fit the verified effective context. Counts carry exact, conservative or unknown quality and provenance. Unknown context or unreliable counting refuses dispatch. Separate context capacity and rate accounting. Drop only complete old conversation turns and preserve the original instructions and current evidence/query bytes. Every permitted replacement must independently admit identical prepared messages and evidence. Reject truncation, capacity failure and incomplete output before publication. Provider stops and client evidence hashes do not prove provider-side context consumption.

Close the maintained exception only after separate activation, reproduced tests-only REDs, independent exact-head review, network-denied acceptance, and a verified GUI merge. The #5/#10/#11/#27 structural regressions, ChromaDB, evidence/search-only rollback and canonical migration gates are preserved. Full #27, #2, the unrelated migration and the separate #27 smoke retain their current scope and gates.

## Verification receipt

The [owner decision and canonical packet](https://github.com/sriharshaguthikonda/easy-local-rag/pull/74) merged as 59fff38ee19fb3329d7db704a4f1b1f35b800a69. The [separate activation](https://github.com/sriharshaguthikonda/easy-local-rag/pull/75) merged as b1aa8ac34a3424de4071321e17ea442900939383. The maintained GUI base was a8926023f8c4ef56ec632137200e6ff9904ca675.

- Tests-only RED commit [43fc47c](https://github.com/sriharshaguthikonda/easy-local-rag/commit/43fc47c7266363610143f5602f3a4937591c1231) reproduced eight admission failures. Tests-only RED [2c46e71](https://github.com/sriharshaguthikonda/easy-local-rag/commit/2c46e716a8afff06e59ca0f468eca4b036a7f220) reproduced four publication and fallback failures. Both preceded the initial product commit.
- Follow-up tests-only RED [860dc6b](https://github.com/sriharshaguthikonda/easy-local-rag/commit/860dc6be2bfb0635f81d09a08c3c4c4f0a6f0512) reproduced two cases where Groq early-stop metadata still produced a structurally accepted answer. The guard in the final head rejects both.
- Three bounded Foreman reviews reused the resident Ornith through the single-slot harness. They confirmed frozen messages and output reserve across fallback, callback suppression on truncation or missing stop, and Groq early-stop rejection on the exact final head. Admission, fallback and stream publication functions matched the earlier reviewed source byte for byte. The wider first attempt timed out and did not count as review.
- A detached checkout at cea1421796f20cf369dc9a5830d52396e9cf747e passed 534 network-denied tests and 11 native synthetic Chroma cases using Python 3.12, ChromaDB 1.0.16 and offscreen Qt. TCP and DNS denial self-checks passed for both runs. Collection retained all 494 baseline test IDs and added 40. Gitleaks and the diff check passed.
- [PR #76](https://github.com/sriharshaguthikonda/easy-local-rag/pull/76) merged into the maintained GUI branch as 435e8f1d7f0212549ab566f5b4c402fadad1d564. The merge tree matched the verified head.

The [deployment guide](https://github.com/sriharshaguthikonda/easy-local-rag/blob/435e8f1d7f0212549ab566f5b4c402fadad1d564/docs/request-admission.md) describes trusted tokenizer and template enrollment. Generation fails closed on unknown deployment metadata. Search and evidence remain available. Verification used synthetic data and transports. Live provider consumption remains unverified. The separate #27 smoke, full #27, #2 and the migration retain their existing gates.
