# Issue #13: Use tiktoken for request budgeting

[Roadmap ledger](../plans/README.md)

**Status:** OPEN — `token_budget.py` and basic unit tests exist, but the legacy request path still uses its regex counter and must be routed to the helper.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/13
**Labels / priority:** `priority:P1`, `type:bug`
**Dependencies:** #16 adds `tiktoken` to runtime requirements; retain a conservative fallback when it is unavailable.

## Implementation slices

1. Use `tiktoken` with a documented default encoding for Groq/Llama requests; keep a conservative word-based fallback.
2. Replace the backend regex counter and budget loop with `count_message_tokens` / `trim_messages_to_budget`.
3. Preserve the system message and current user question; trim older turns and oversized retrieved context first, with a visible trim warning.

## Affected interfaces, files, and artifacts

- `token_budget.py`, `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `streamlit_app.py`, `requirements.txt`, tests.
- Request budget contract: explicit context limit, input ceiling, output ceiling, and safety margin.

## Concrete actions

- Audit each API request builder; no live request should use the regex estimate as its primary counter.
- Do not silently delete the current query; record that context/history was trimmed.
- Add a code-heavy text test and ordering tests for system/current-message preservation.

## Verification

```powershell
python -m pytest tests/test_token_budget.py -q
python -m pytest tests -q
python -m py_compile token_budget.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** every maintained Groq request uses the shared budget helper, regex is not primary, conservative fallback works, and trim ordering is tested.
- **Retirement closure (mutually exclusive):** remove uncovered Groq request builders from supported entry points and docs, prove they cannot issue requests, and identify the maintained budgeted replacement. Retirement cannot leave an unchecked request route.
- **Rollback constraint:** retain a conservative fallback; never restore unchecked requests solely to avoid a dependency error.
- **Commit:** `fix(#13): use tiktoken for Groq budgeting`.
