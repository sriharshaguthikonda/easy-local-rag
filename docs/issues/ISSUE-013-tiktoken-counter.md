# Issue #13 request admission for maintained ChromaDB

[Roadmap ledger](README.md) · [GitHub issue](https://github.com/sriharshaguthikonda/easy-local-rag/issues/13)

**Status:** OPEN. The owner approved a one-time maintained ChromaDB exception on 28 November 2026 while #2 remains open. The [pinned audit](../Research/issue-013-token-context-audit.md) and [canonical implementation packet](../superpowers/plans/2026-11-28-issue-013-request-admission-implementation.md) define the verification, target and activation gate.

All maintained request entry points use candidate-specific admission. The complete request and reserved output must fit the verified effective context. Counts carry exact, conservative or unknown quality and provenance. Unknown context or unreliable counting refuses dispatch. Separate context capacity and rate accounting. Drop only complete old conversation turns and preserve the original instructions and current evidence/query bytes. Every permitted replacement must independently admit identical prepared messages and evidence. Reject truncation, capacity failure and incomplete output before publication. Provider stops and client evidence hashes do not prove provider-side context consumption.

Close the maintained exception only after separate activation, reproduced tests-only REDs, independent exact-head review, network-denied acceptance, and a verified GUI merge. The #5/#10/#11/#27 structural regressions, ChromaDB, evidence/search-only rollback and canonical migration gates are preserved. Full #27, #2, the unrelated migration and the separate #27 smoke retain their current scope and gates.
