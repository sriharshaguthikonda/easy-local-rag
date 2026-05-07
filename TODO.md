## Open audit issues (P0/P1 — cross-repo audit 2026-04-26)

Branch: `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`. Source of truth: [GitHub Issues](https://github.com/sriharshaguthikonda/easy-local-rag/issues).

### P0 — Critical
- [#2](https://github.com/sriharshaguthikonda/easy-local-rag/issues/2) Rotate and remove hardcoded Groq API keys committed to repo — `type:security`
- [#6](https://github.com/sriharshaguthikonda/easy-local-rag/issues/6) Hybrid retrieval crashes on missing meta[embedding] key in MMR loop — `type:bug`
- [#9](https://github.com/sriharshaguthikonda/easy-local-rag/issues/9) Validate keys on streamlit conversation import to prevent state hijack — `type:security`
- [#15](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15) Atomic-write vault.json to stop double-append corruption — `type:bug`

### P1 — Important
- [#5](https://github.com/sriharshaguthikonda/easy-local-rag/issues/5) Mitigate prompt injection from retrieved documents — `type:security`
- [#7](https://github.com/sriharshaguthikonda/easy-local-rag/issues/7) Lazy-init ChromaDB collection so fresh users can launch the app — `type:bug`
- [#8](https://github.com/sriharshaguthikonda/easy-local-rag/issues/8) Replace hardcoded Windows-only paths with env/config-driven values — `type:bug`
- [#10](https://github.com/sriharshaguthikonda/easy-local-rag/issues/10) Build BM25 index over full corpus, not top-50 vector hits — `type:perf`
- [#11](https://github.com/sriharshaguthikonda/easy-local-rag/issues/11) Enforce embedding model match between ingest and query — `type:bug`
- [#12](https://github.com/sriharshaguthikonda/easy-local-rag/issues/12) Fix TTS worker thread / asyncio deadlock in streamlit re-runs — `type:bug`
- [#13](https://github.com/sriharshaguthikonda/easy-local-rag/issues/13) Replace regex token counter with tiktoken to avoid Groq rate-limit hits — `type:bug`
- [#14](https://github.com/sriharshaguthikonda/easy-local-rag/issues/14) Render inline citations [N] tying answer claims to source chunks — `type:enhancement`
- [#16](https://github.com/sriharshaguthikonda/easy-local-rag/issues/16) Add .env.example, AGENTS.md, complete requirements lock — `type:dx`
