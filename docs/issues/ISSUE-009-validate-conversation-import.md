# Issue #9: Validate imported Streamlit conversations

**Status:** OPEN — whitelist helper and basic tests exist; this is the next small closure after missing regression coverage.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/9
**Labels / priority:** `priority:P0`, `type:security`
**Dependencies:** Do not import #4-style source paths; source state is reconstructed from retrieval.

## Implementation slices

1. Keep `conversation_import.py` as the pure whitelist boundary: only `history`, `tags`, and `favorites`; reject underscore-prefixed and all runtime-object keys.
2. Assign validated fields explicitly in `streamlit_app.py`; display rejected-key/type warnings.
3. Complete missing tests for crafted `current_sources`, oversized content, valid old exports, and preservation of the real collection/client state.

## Affected interfaces, files, and artifacts

- `conversation_import.py`, Streamlit import handler, `tests/test_conversation_import.py`.
- Import format: JSON object with validated messages (`system`, `user`, `assistant`), string content under a documented size limit, safe tags/favorites only.

## Concrete actions

- Never call `st.session_state.update(imported_data)`.
- Validate nested containers before assignment; ignore unknown values with user-visible warnings rather than trusting them.
- Test through the helper and the Streamlit integration boundary without needing a live Chroma client.

## Verification

```powershell
python -m pytest tests/test_conversation_import.py -q
python -m pytest tests -q
python -m py_compile conversation_import.py streamlit_app.py
```

Manual: import JSON containing `collection`, `tts_worker`, and `current_sources`; confirm warnings appear and the real session objects remain unchanged.

## Closure gate, rollback, and commit boundary

- **Close only when:** malicious state keys, unsafe source paths, invalid roles, and oversized content are rejected; legacy valid history imports; warnings are shown.
- **Rollback constraint:** retain explicit assignments and the whitelist; never restore bulk session-state update.
- **Commit:** `fix(#9): validate conversation imports`.
