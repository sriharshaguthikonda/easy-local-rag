# Issue #12: Stabilize the Streamlit TTS worker

**Status:** OPEN — `@st.cache_resource` appears in Streamlit, but the legacy backend still contains ad hoc threads and repeated event-loop startup.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/12
**Labels / priority:** `priority:P1`, `type:bug`
**Dependencies:** Preserve the TTS-disabled fast path; no new audio dependency is required.

## Implementation slices

1. Use one managed queue and one background worker per cached resource/session.
2. Run asynchronous TTS with one worker-owned event loop, or make it synchronous inside that worker; do not call `asyncio.run()` per sentence in a long-lived worker.
3. Add stop sentinel, idempotent start, and rerun/death recovery tests.

## Affected interfaces, files, and artifacts

- `streamlit_app.py`, `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `kokoro_tts.py`, optional `tts_worker.py`, tests.
- Runtime contract: no duplicate playback, deadlock, or leaked test thread over repeated reruns.

## Concrete actions

- Trace all thread creation sites; remove or isolate legacy global workers from the Streamlit path.
- Cache the worker resource, store only simple flags in session state, and provide clean shutdown.
- Add a small fake synthesis test rather than calling a live audio provider.

## Verification

```powershell
python -m pytest tests/test_streamlit_tts_worker_config.py -q
python -m pytest tests -q
python -m py_compile streamlit_app.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

Manual: submit three prompts with TTS on, toggle it off/on, and confirm one audio stream per answer with no deadlock.

## Closure gate, rollback, and commit boundary

- **Close only when:** start is idempotent, stop is clean, reruns reuse one worker, and manual toggling produces no duplicate audio.
- **Rollback constraint:** retain a safe disabled-TTS path; do not restore unbounded daemon-thread creation.
- **Commit:** `fix(#12): stabilize Streamlit TTS worker`.
