# Issue #12: Stabilize the Streamlit TTS worker

[Roadmap ledger](../plans/README.md)

**Status:** OPEN — `@st.cache_resource` appears in Streamlit, but the legacy backend still contains ad hoc threads and repeated event-loop startup.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/12
**Labels / priority:** `priority:P1`, `type:bug`
**Dependencies:** Preserve the TTS-disabled fast path; no new audio dependency is required.

## Implementation slices

1. Give each Streamlit session an owned queue, worker, and routing/session ID; never route audio through a process-global queue shared by sessions.
2. Run asynchronous TTS with one worker-owned event loop, or make it synchronous inside that worker; do not call `asyncio.run()` per sentence in a long-lived worker.
3. Add a session-owned stop sentinel, idempotent start, rerun/death recovery, and shutdown that joins only that session's worker.
4. Run two sessions concurrently with distinct fake synthesis payloads; assert no cross-session audio, queue consumption, stop, or shutdown interference.

## Affected interfaces, files, and artifacts

- `streamlit_app.py`, `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `kokoro_tts.py`, optional `tts_worker.py`, tests.
- Runtime contract: no duplicate playback, deadlock, or leaked test thread over repeated reruns.

## Concrete actions

- Trace all thread creation sites; remove or isolate legacy global workers from the Streamlit path.
- Cache or own the worker at session scope, store only simple routing/lifecycle state in session state, and provide clean session shutdown.
- Add a small fake synthesis test rather than calling a live audio provider.

## Verification

```powershell
python -m pytest tests/test_streamlit_tts_worker_config.py -q
python -m pytest tests -q
python -m py_compile streamlit_app.py streamlit_groq_lama_chromadb_RAG_ETTS.py
```

Manual: submit three prompts with TTS on, toggle it off/on, and confirm one audio stream per answer with no deadlock.

## Closure gate, rollback, and commit boundary

- **Maintained-path closure:** start is idempotent, stop/shutdown are session-owned and clean, reruns reuse that session's worker, the concurrent two-session isolation regression passes, and manual toggling produces no duplicate audio.
- **Retirement closure (mutually exclusive):** remove TTS from maintained Streamlit controls and setup docs, prove no worker/queue starts from that path, and document any maintained TTS replacement with its own session isolation evidence.
- **Rollback constraint:** retain a safe disabled-TTS path; do not restore unbounded daemon-thread creation.
- **Commit:** `fix(#12): stabilize Streamlit TTS worker`.
