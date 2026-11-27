# Issue #5 — Treat retrieved documents as untrusted prompt context

[Roadmap ledger](README.md) · [Implementation packet](../superpowers/plans/2026-11-27-issue-005-guarded-requests-implementation.md)

**Status:** CLOSED — Verified and merged through PR #66.
**GitHub** https://github.com/sriharshaguthikonda/easy-local-rag/issues/5
**Labels** `priority:P1`, `type:security`
**Authority** [Owner approval](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5#issuecomment-7213652317).
**LM Studio smoke activation** The reviewed [packet amendment PR #67](https://github.com/sriharshaguthikonda/easy-local-rag/pull/67) merged as `44fc82647cb2f6d0651240106c1e24257c03154b`. This separate docs-only activation authorizes the synthetic smoke at PR #66 head `27959f498b95ca1095f8fff4ec93d50c7c0b2e81` under the amended packet.
**Frozen GUI base** `89514bf502a39776b2261ff1657146f530364765`, which includes the verified #10 and #11 implementations.

## Scope and invariants

- The exception covers the PyQt answer worker, Streamlit web synchronous and background paths, and both maintained ChromaDB CLIs for Groq and Ollama. These entry points share the guarded request helper in `rag_prompting.py`.
- Keep application policy and the evidence-handling guard in the system role. Treat retrieved strings and metadata as evidence. Serialize them reversibly inside source wrappers, with trusted roles and wrapper syntax controlled by application code. Keep exact source text, order, citation numbering, and the `(collection_id, chunk_id)` pair. Preserve the canonical Streamlit citation and single-evidence-source contract.
- Keep evidence in the current request. Persist the original question and accepted assistant answer as the durable turns. Preserve provider selection and bounded fallback with an identical prepared message sequence.
- RAG answer generation accepts text only. Document text and provider tool events have no action authority. Cloud speech requires separate explicit opt-in, including with local model selection. Untrusted rich text must not cause automatic resource loads. Explicit user actions and transfers to the selected generation provider retain their existing authority.
- Retain ChromaDB, offline lexical search and all verified #10 and #11 guarantees. Use synthetic data and keep corpus storage, credentials, historical refs and #2's existing gate outside the exception.
- Full #27 claim validation, #22's complete provider abstraction, #25 retirement, PostgreSQL migration and MCP integration retain their own scope and sequence.

## Execution and closure

1. Merge the independently reviewed locked packet, then merge a separate activation PR before any product edit.
2. On the frozen tree, commit and record tests-only RED evidence for raw CLI text, wrapper serialization, misplaced policy, evidence replay, handled tool output and automatic egress. Use harmless data and fake providers and actions. Existing safe behavior is recorded as a passing control.
3. Verify focused tests, the complete regression suite and the native synthetic Chroma #10/#11 cases with provider networking denied. Preserve both retrieval identities and unchanged trusted roles. Review the exact candidate independently and repeat verification from a separate clean checkout before merge. Any change to the candidate or frozen GUI base invalidates the corresponding evidence.
4. Run the synthetic model smoke in the packet only under the applicable provider and credential gates. The owner-approved 2026-11-27 packet amendment also permits the already-loaded, manifest-verified Ornith GGUF on the existing LM Studio loopback endpoint for the exact-head synthetic rerun. All other acceptance criteria continue to apply. Record the actual local model and observed behavior with inert synthetic evidence. Request captures prove structure and host-side permissions. Model refusal remains model-dependent. If a permitted smoke cannot run or does not pass, keep #5 open.
5. Publish the RED, focused, full, native, independent review and model-smoke receipts with the exact reviewed and merged SHAs. Close #5 after its amended criteria pass and reconcile the ledger.

## Closure evidence

- [PR #66](https://github.com/sriharshaguthikonda/easy-local-rag/pull/66) merged into the maintained GUI branch as `656ae2a905cb4a400e66a9101663de546b2d17e0` from independently reviewed head `27959f498b95ca1095f8fff4ec93d50c7c0b2e81`. The tests-only commits recorded 38 expected failures. The separate checkout passed 140 focused tests, 102 Chroma and offline regressions, 11 direct native synthetic Chroma cases, and all 395 tests with provider networking denied. Compilation and GitHub gitleaks also passed. [Exact-head verification](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5#issuecomment-7214809517)
- After the [reviewed amendment #67](https://github.com/sriharshaguthikonda/easy-local-rag/pull/67) and [separate activation #68](https://github.com/sriharshaguthikonda/easy-local-rag/pull/68) merged, the already-loaded, manifest-verified Ornith GGUF returned `blue` for both exact-head synthetic fixtures through LM Studio. Both composer message hashes were reproduced independently. The existing Foreman harness passed its post-smoke check. [Behavioral receipt](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5#issuecomment-7218099584)
- Native Ollama router and request verification is in the exact-head regression evidence. The earlier native Ollama Qwen 0.5B behavioral smoke returned `ORANGE` on the adversarial fixture, including on the stricter in-memory rerun. The LM Studio result satisfies the amended behavioral gate for the tested head and fixtures. [Native Ollama and Qwen record](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5#issuecomment-7214809517)
- The approved criteria passed and #5 closed. #2 stays open under its existing gate. [Closure receipt](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5#issuecomment-7218274040)

## Rollback constraint

Use a reviewed GUI follow-up that preserves the guard and app-controlled action permissions. Raw concatenation, unreviewed disclosure and ingest-time text rewriting are outside the rollback path.
