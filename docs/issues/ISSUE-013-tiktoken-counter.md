# Issue #13 Plan: Replace regex token counter with tiktoken

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/13

Priority: P1 bug

## Goal

Use a real tokenizer estimate for request budgeting so Groq requests do not
regularly exceed model limits and fall into broken fallback behavior.

## Files to inspect

- `streamlit_groq_lama_chromadb_RAG_ETTS.py`
- `streamlit_app.py`
- `requirements.txt`
- optional new file: `token_budget.py`
- tests under `tests/`

## Implementation steps

1. Add `tiktoken` to `requirements.txt` in issue #16 or in this commit if issue
   #16 has not landed yet.
2. Add a helper module, for example `token_budget.py`.
3. In the helper, implement:

   - `encoding_for_model(model_name)`
   - `count_message_tokens(messages, model_name)`
   - `trim_messages_to_budget(messages, max_input_tokens)`

4. Use `tiktoken.get_encoding("cl100k_base")` as the default for Groq Llama
   models unless a better mapping is proven.
5. Keep a fallback if `tiktoken` is unavailable, but make it conservative:

   ```python
   estimated = int(len(text.split()) * 1.6)
   ```

6. Replace the regex tokenizer in
   `streamlit_groq_lama_chromadb_RAG_ETTS.py::count_tokens`.
7. Set budgets explicitly:

   - model context limit
   - max input tokens
   - max output tokens
   - safety margin, for example 70 percent of Groq TPM when TPM is the limiting
     factor

8. When trimming, remove oldest user/assistant turns first. Keep the system
   message.
9. If a single retrieved context chunk is too large, truncate context before
   truncating the current user query.
10. Log or show a small warning when context was trimmed.

## Tests and verification

Add tests:

- token count for code text is higher than naive word count
- system message is preserved during trimming
- oldest chat turn is removed first
- huge context is trimmed before current query

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile token_budget.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

Manual smoke:

1. Create a long retrieved context.
2. Send a Groq request.
3. Expected: request is trimmed before API call.
4. Expected: no 413/429 caused by obvious token undercount.

## Acceptance checklist

- [ ] Regex tokenizer is not the primary token counter.
- [ ] `tiktoken` is listed in requirements.
- [ ] Token budget has a safety margin.
- [ ] Trimming preserves the system message and current query.
- [ ] Tests cover token counting and trimming.

## Commit boundary

Use one commit for this issue only:

```text
fix(#13): use tiktoken for Groq budgeting
```
