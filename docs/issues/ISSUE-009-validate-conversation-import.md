# Issue #9: Validate imported Streamlit conversations

**Status:** OPEN — whitelist helper and basic tests exist; this is the next small closure after missing regression coverage.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/9
**Labels / priority:** `priority:P0`, `type:security`
**Dependencies:** Do not import #4-style source paths; source state is reconstructed from retrieval.

## Implementation slices

1. Keep `conversation_import.py` as the pure whitelist boundary: only `history`, `tags`, and `favorites`; reject underscore-prefixed and all runtime-object keys. Enforce an 8 MiB file limit, 1,000-message limit, 64 KiB per-message content limit, 100 tags, 100 favorites, and 256 characters per tag/favorite before allocating complete nested state.
2. Assign validated fields explicitly in `streamlit_app.py`; display rejected-key/type warnings.
3. Complete missing tests for crafted `current_sources`, each exact limit and one-over rejection, valid old exports, and preservation of the real collection/client state.

## Affected interfaces, files, and artifacts

- `conversation_import.py`, Streamlit import handler, `tests/test_conversation_import.py`.
- Import format: JSON object with validated messages (`system`, `user`, `assistant`), string content under a documented size limit, safe tags/favorites only.

## Concrete actions

- Never call `st.session_state.update(imported_data)`.
- Check raw byte size before JSON parsing; validate container counts and item sizes before building replacement lists, and complete validation before assigning any session-state field.
- Reject over-limit imports atomically with a user-visible warning; do not partially import a prefix.
- Test through the helper and the Streamlit integration boundary without needing a live Chroma client.

## Verification

```powershell
python -m pytest tests/test_conversation_import.py -q
python -m pytest tests -q
python -m py_compile conversation_import.py streamlit_app.py
```

Manual: import JSON containing `collection`, `tts_worker`, and `current_sources`; confirm warnings appear and the real session objects remain unchanged.

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** malicious state keys, unsafe source paths, invalid roles, and exact-limit violations are rejected before full state allocation or any assignment; valid legacy history imports; warnings are shown.
- **Retirement closure (mutually exclusive):** remove conversation import from every maintained UI and setup document, prove no upload/import handler reaches JSON-to-session assignment, and document the supported replacement/export workflow. Retirement does not permit a hidden unsafe import route.
- **Rollback constraint:** retain explicit assignments and the whitelist; never restore bulk session-state update.
- **Commit:** `fix(#9): validate conversation imports`.
