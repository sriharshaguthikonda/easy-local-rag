# Issue #9 Plan: Validate Streamlit conversation imports

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/9

Priority: P0 security

## Goal

Importing a conversation JSON must not blindly overwrite `st.session_state`.
Only safe keys with validated types should be accepted.

## Files to inspect

- `streamlit_app.py`
- optional new file: `conversation_import.py`
- tests under `tests/`

## Safe import contract

Allowed top-level keys:

- `history`: list of chat messages
- `tags`: dict of string keys to string/list values
- `favorites`: list of saved response objects or strings

Do not import:

- `chroma_client`
- `collection`
- `current_sources`
- `source_filters`
- `tts_queue`
- `tts_worker`
- any key beginning with `_`

## Implementation steps

1. Add a pure helper, for example `conversation_import.py`.
2. Implement `validate_message(message)`:

   - must be a dict
   - `role` must be one of `system`, `user`, `assistant`
   - `content` must be a string under a reasonable size limit

3. Implement `sanitize_conversation_import(data)`:

   - parse only dict input
   - copy allowed keys into a new dict
   - validate every value
   - ignore unknown keys or return them as rejected warnings

4. In `streamlit_app.py`, replace:

   ```python
   st.session_state.update(imported_data)
   ```

   with explicit assignments:

   ```python
   safe_data, warnings = sanitize_conversation_import(imported_data)
   st.session_state.conversation_history = safe_data.get("history", [])
   st.session_state.tags = safe_data.get("tags", {})
   st.session_state.favorite_responses = safe_data.get("favorites", [])
   ```

5. Show a warning in the UI when unknown or invalid keys are rejected.
6. Keep export format backward compatible by either continuing to export
   `history`, `tags`, `favorites`, or adding a `version` field while still
   accepting old files.
7. Never accept imported file paths into `current_sources`. Sources must be
   rebuilt from retrieval.

## Tests and verification

Add tests:

- malicious import with `{"collection": "...", "tts_worker": "..."}`
- malicious import with `current_sources` file path
- valid minimal history imports
- invalid message role is rejected
- huge content is rejected or truncated with a warning

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile conversation_import.py streamlit_app.py
```

Manual smoke:

1. Import JSON containing `{"collection": "owned", "history": []}`.
2. Expected: UI warns that `collection` was rejected.
3. Expected: `st.session_state.collection` remains the real Chroma collection.

## Acceptance checklist

- [ ] Import uses a whitelist, not `st.session_state.update`.
- [ ] Unsafe session keys are rejected.
- [ ] Valid exports still import.
- [ ] Rejected keys are visible to the user.
- [ ] Tests cover malicious state overwrite.

## Commit boundary

Use one commit for this issue only:

```text
fix(#9): validate conversation imports
```
