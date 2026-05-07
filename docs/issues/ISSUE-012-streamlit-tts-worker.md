# Issue #12 Plan: Fix Streamlit TTS worker deadlock

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/12

Priority: P1 bug

## Goal

Make TTS processing stable across Streamlit reruns. There should be one managed
worker resource per app session, not a new ad hoc thread/event-loop combination
that can deadlock or duplicate audio.

## Files to inspect

- `streamlit_app.py`
- `streamlit_groq_lama_chromadb_RAG_ETTS.py`
- `kokoro_tts.py`
- tests under `tests/`

## Implementation steps

1. Extract TTS worker management into a small module, for example
   `tts_worker.py`.
2. In that module, define a class with:

   - one `queue.Queue`
   - one background thread
   - one event loop created inside that thread if async TTS is required
   - `start()`
   - `enqueue(text)`
   - `stop()`

3. Do not call `asyncio.run()` once per sentence inside a long-lived worker
   thread. Use one event loop in the worker or make TTS fully synchronous inside
   that thread.
4. In `streamlit_app.py`, create the worker using `@st.cache_resource`:

   ```python
   @st.cache_resource
   def get_tts_worker():
       worker = TTSWorker(...)
       worker.start()
       return worker
   ```

5. Store only lightweight flags in `st.session_state`, not raw worker internals.
6. Make reruns idempotent:

   - if worker exists and alive, reuse it
   - if worker died, create a new one and show a warning

7. Add a stop sentinel for shutdown. Do not leak daemon threads during tests.
8. Keep TTS disabled path cheap. If `tts_enabled` is false, do not enqueue.
9. If `nest_asyncio.apply()` is only needed for old code, remove it from this
   path or isolate it away from the worker.

## Tests and verification

Add tests for pure worker behavior:

- calling `start()` twice creates one thread
- enqueue sends one item
- `stop()` drains or stops cleanly
- simulated rerun reuses cached worker function

Suggested commands:

```powershell
python -m pytest tests -q
python -m py_compile tts_worker.py streamlit_app.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

Manual smoke:

1. Run `streamlit run streamlit_app.py`.
2. Send 3 prompts with TTS enabled.
3. Toggle TTS off and on.
4. Expected: no duplicated audio and no deadlock after reruns.

## Acceptance checklist

- [ ] TTS worker is managed as a cached resource.
- [ ] Reruns do not spawn duplicate workers.
- [ ] Worker uses one event loop or synchronous TTS, not repeated `asyncio.run`.
- [ ] Stop/shutdown path exists.
- [ ] Tests cover idempotent worker startup.

## Commit boundary

Use one commit for this issue only:

```text
fix(#12): stabilize Streamlit TTS worker
```
