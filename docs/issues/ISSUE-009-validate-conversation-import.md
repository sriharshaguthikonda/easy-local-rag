# Issue #9: Validate imported Streamlit conversations

[Roadmap ledger](README.md) · [implementation packet](../superpowers/plans/2026-07-24-issue-009-validate-conversation-import-implementation.md)

**Status:** PLANNING during packet review; no implementation is active.

**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/9
**Priority/type:** `priority:P0`, `type:security`

## Canonical acceptance contract

Maintain the existing exporter schema (`history`, `sources`, `tags`, `favorites`) while making imports a raw-byte atomic whitelist boundary. Only UTF-8 JSON objects at most 8 MiB are accepted. Allowed top-level keys are `history`, `tags`, `favorites`, and legacy export-only `sources`; the latter is ignored with exactly `Ignored legacy export-only key: sources.`. All other keys reject, including trusted runtime/session keys and underscore keys.

History accepts at most 1,000 messages with exactly `role`/`content`, roles `system`/`user`/`assistant`, and UTF-8 content at most 64 KiB. JSON container depth is at most 32, with the root object as depth 1 and every nested dict/list adding one; parser recursion is a rejection. Tags accept at most 100 string keys and 100 string values total, each tag text at most 256 characters. Favorites accept at most 100 shallow dictionaries, with string keys and scalar string/int/finite-float/bool/null values; every favorite string key/value is at most 256 characters. `NaN`, infinities, and numeric overflow (for example `1e999`) reject globally; bool is accepted only as a favorite scalar.

Validation completes before any state write. Validated fields map only to `conversation_history`, `tags`, and `favorite_responses`; `sources`, `current_sources`, collection/client/filter/TTS runtime objects, and all other runtime keys are never assigned. Rejection performs zero writes and UI shows deterministic `Conversation import rejected: {error}`.

## Execution and closure

The packet locks the frozen GUI code base and executable test/implementation snippets. The lifecycle is: planner packet -> ChatGPT review -> GSD checker -> corrector -> docs merge -> implementer initial TDD commit -> code-review agent -> accepted-finding fixer commit(s) -> verifier -> orchestrator PR merge/evidence/close. ChatGPT reviewed `8d3f8d1e344e607235c362d46a589d313c1ee28d` with blockers; the packet ledger records their dispositions and requires a fresh exact-head re-review without copying the earlier outcome forward. Before docs merge/implementation and code-PR merge/closure, the orchestrator confirms the standing Q&A authorization remains unrevoked and all named review/evidence gates pass; otherwise: **STOP; do not proceed**. No new per-packet reply is needed while that standing authorization remains current.

The code branch is `codex/fix-issue-9` directly from `daecce8a27f50da39284f5519d77b835905209f6`, never merged/rebased with main docs. Its initial commit is `fix(#9): validate conversation imports`; closure names all docs/code/review-fix/final SHAs, compares frozen/post-change pytest/compile/live-issue commands by exit code, summary, and failing test ID, and allows no new failure. Rollback is revert of that commit and accepted review-fix commits in reverse order.
