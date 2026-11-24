# Public branch and PR inventory

Snapshot: 2026-08-02 06:13:17 UTC. This is the issue #18 pre-rewrite
inventory. It intentionally omits credential values and private corpus
content. The exact manifest and redacted scanner report are retained outside
the repository with user-only ACLs. Re-fetch before any later ref operation;
this document does not authorize deletion, rewriting, archive publication, or
merging.

## Evidence contract

- Fetch covered remote heads, tags, and PR heads through explicit refspecs.
- Exact private manifest: 34 local heads, 32 remote heads (31 non-default), 15
  PR refs, and zero tags. Manifest SHA-256:
  `fd7ca3b8dae52277d38d9c44a135ceebd15d5033d6f8891572208d4fe21e4862`.
- Gitleaks `8.30.1` ran over `--all` with `.gitleaks.toml` SHA-256
  `7d863dffa2d7d8520f76de1c8013eed10676b718a6c62a66616157c6c68abf86`.
  Exit `3` and 13 redacted findings agree, which is a valid pre-rewrite
  inventory result. Private report SHA-256:
  `fdb51ad92f27397327ac1d48160fc8760c38075338eb7695abcc6b2291a0f247`.
- Finding locations only: 12 `groq-api-key` findings span two historical
  commits (`214d4216...`, `dd75fa5a...`) and six paths; one
  `generic-api-key` finding is in `lib/vis-9.1.2/vis-network.min.js` at
  `1ea8b487...`. The inventory and scanner report are redacted; exposed values remain in shared history and are not reproduced here.
- The Groq findings are reachable from fetched PR refs #1, #28-#36, and
  #39-#43. The generic finding is reachable from #1, #29, #33, #36, #42, and
  #43. #26B must reconcile those immutable refs before any rewrite outcome is
  accepted.

`Behind` and `ahead` below are commit counts from `origin/main...<tip>`;
`paths` is the triple-dot changed-path count. Exact SHAs, not counts, are the
inventory identity.

## Remote heads

| Remote branch | Immutable tip | Behind | Ahead | Paths |
|---|---:|---:|---:|---:|
| `B25-hyb-kro-tkn-lmt-synms-mon-chngs-strmlit-chrdb-docs-web` | `355cf8560470b8752b5962da083d1599b9a9807c` | 60 | 17 | 35 |
| `BM25-hyb-gtts-tkn-lmt-synms-mon-chngs-streamlit` | `46f81c03b6794656f47aa354c2562fbc78cd8725` | 60 | 11 | 27 |
| `BM25-hyb-gtts-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` | `0a471504f44e56bc026d7f1ee1e7cf46619f714a` | 60 | 13 | 27 |
| `BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` | `ed662570fcd50beb32fc6535bcbd421465f7f881` | 60 | 21 | 67 |
| `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` | `f78d53476a452dd66aa207d74ebd7d2dca458fca` | 60 | 91 | 133 |
| `Pubmed_bert` | `3ceb6c2f4a04f4dd97e5e1bea674af1ee97408e6` | 136 | 16 | 10 |
| `agent/canonical-migration-plans` | `b11ee66f9c8e6e3eaa5fe388adeeaebe7ec04039` | 60 | 4 | 4 |
| `codex/add-qtpy-gui-for-groq_lama_chromadb_rag_etts.py` | `8cb08091347ee41e7c8dac1c04d02a8e5d2eed2b` | 60 | 16 | 33 |
| `codex/fix-issue-15` | `3ced61bcb9957907cc568e2e2560edd6f3c53b83` | 60 | 72 | 105 |
| `codex/fix-issue-37` | `ef0d4d48b022b013e9dce75d9aaf62f0ddf3bf71` | 1 | 0 | 0 |
| `codex/fix-issue-6` | `de7465902e00e363993f9ea9eb2190ce940e1bed` | 60 | 55 | 105 |
| `codex/fix-issue-9` | `9e77fcf628eb9caaf563da133e4186930769eb61` | 60 | 58 | 105 |
| `codex/github-issue-plans` | `bcb0de4c3382ba8b3630d05f82e2a9fc20b56af3` | 49 | 0 | 0 |
| `codex/issue-009-activate` | `447d070cdfd6b99d68c76e003f1a8803467d264d` | 33 | 0 | 0 |
| `codex/issue-009-plan` | `3b8f173501fea20fed3f35421886be503c7fdfec` | 35 | 0 | 0 |
| `codex/issue-015-activate` | `13e18611b5db9cc4881b85a266d8259107641719` | 26 | 0 | 0 |
| `codex/issue-015-plan` | `14df1b7fcea0775c56b6a774c4047c3f14e29c9d` | 28 | 0 | 0 |
| `codex/issue-037-activate` | `ad46b0c878d99e1639f6bed7590162ab3b173d93` | 14 | 0 | 0 |
| `codex/issue-037-plan` | `cba86f0e1f6cd2d03c6280ebda88e8714925af9a` | 16 | 0 | 0 |
| `codex/issue-038-closeout` | `90a541c00b3809a1190c2820e883266c40133f60` | 60 | 90 | 133 |
| `codex/issue-038-evidence-contract` | `5e5e8941670f519e6447a770fe0f0d012b96cbcc` | 60 | 87 | 133 |
| `codex/qwen-embedding-migration-plan` | `53824420ea588fdcbd265b5a0181308a1cff314f` | 60 | 55 | 106 |
| `cos_chroma` | `c47efb53c69d1a64d40b12e025a96166b7e33935` | 136 | 1 | 5 |
| `hyb-gtts-tkn-lmt-synms-monitor-chngs` | `cb1a793e4f7861af919204e57d41bdac69f91352` | 60 | 6 | 4 |
| `hyb-gtts-tkn-lmt-synonyms` | `ec16d7ce0c6bb58b881a6016fa83ea17b3835bdd` | 60 | 6 | 14 |
| `hybrid-coquiTTS` | `85ab9e380cbd6fa0c457830acd3bc1bdf5ba32c8` | 60 | 4 | 2 |
| `hybrid-gtts-tkn-lmt` | `3c802d6336ce060c0af977e22b9ef8dc61ed71ad` | 60 | 4 | 2 |
| `hybrid_search` | `54f25f4cd863e1bba6a8e986b632fa498f4d8d99` | 60 | 2 | 2 |
| `main` | `f87b4cfcff4351c8f6e1e73ef43a7bf99e7fd5f9` | 0 | 0 | 0 |
| `media_keys` | `f5e9d03f8c01c87315f53a027882927600fa1f15` | 61 | 1 | 4 |
| `streamlit` | `b2992e37d55ceda26e630e8cc5be6fed44478dba` | 60 | 9 | 20 |
| `streamlit-mon-seq-align` | `728123f102465843f46f87121e15d14d4008da0f` | 60 | 9 | 23 |

## Local-only heads

| Local branch | Immutable tip | Behind | Ahead | Paths | Proposed disposition |
|---|---:|---:|---:|---:|---|
| `Pubmed_Bert_cos` | `e0b2ac7c94589675dfbc8abebb3f7afec416f8dd` | 136 | 6 | 7 | inspect with PubMed family; preserve until #26B |
| `codex/issue-015-close` | `acc36e435ca5748c37469c6a33732b7911d91c3f` | 25 | 2 | 2 | merged closure evidence; preserve until #26B |
| `hybrid-gtts` | `b5e3d244143fcc507b2f679740db2d44390d63a1` | 60 | 3 | 2 | inspect with hybrid family; preserve until #26B |
| `simplifyDB` | `2247a37fe2aa67fdc8d365c3b259958e160cffd5` | 136 | 0 | 0 | no unique commits vs `main`; preserve until #26B |

## Local/remote name divergences

These local names also exist remotely but do not point at the same object, so
both tips must be reconciled before either name is changed. `Remote only` and
`local only` are commit counts between the named tips; `n/a` means no merge
base. The current working-head row includes the Step 1 inventory commit.

| Branch | Local tip | Remote tip | Remote only | Local only | Disposition |
|---|---:|---:|---:|---:|---|
| `BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` | `e4808b636ea1103702bf39a480fd21c01c5de876` | `ed662570fcd50beb32fc6535bcbd421465f7f881` | 5 | 0 | local is an old ancestor; remote tip is authoritative inventory |
| `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` | `38e863757e76ed7a321584994282c52585dfa941` | `f78d53476a452dd66aa207d74ebd7d2dca458fca` | 37 | 0 | local is an old ancestor; remote tip is current GUI lineage |
| `codex/issue-038-evidence-contract` | `12f1f9be04e820ae50b071ab91758dbb78ee59de` | `5e5e8941670f519e6447a770fe0f0d012b96cbcc` | 0 | 2 | preserve local #2A/#18 work for integration |
| `codex/qwen-embedding-migration-plan` | `c371c2c2c2cf7de57845ff2a036b6fe26aa6691b` | `53824420ea588fdcbd265b5a0181308a1cff314f` | 0 | 64 | combined-history local proposal; retain canonical docs only after comparison |
| `cos_chroma` | `cee61eeec72e16ac5ccd4e837071f5a68fb983e7` | `c47efb53c69d1a64d40b12e025a96166b7e33935` | n/a | n/a | unrelated commits with identical tree; retain both SHAs until #26B |
| `main` | `3e9e864a73137190dd202bdf4827e7a86f31aa84` | `f87b4cfcff4351c8f6e1e73ef43a7bf99e7fd5f9` | 60 | 0 | local is an old ancestor; do not use as execution authority |

## Pull requests and fetched PR heads

| PR | State | Base | Head | Immutable head |
|---:|---|---|---|---:|
| #1 | closed, unmerged | `BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` | `codex/add-qtpy-gui-for-groq_lama_chromadb_rag_etts.py` | `8cb08091347ee41e7c8dac1c04d02a8e5d2eed2b` |
| #28 | merged | `main` | `agent/canonical-migration-plans` | `b11ee66f9c8e6e3eaa5fe388adeeaebe7ec04039` |
| #29 | merged | GUI branch | `codex/fix-issue-6` | `de7465902e00e363993f9ea9eb2190ce940e1bed` |
| #30 | merged | `main` | `codex/github-issue-plans` | `bcb0de4c3382ba8b3630d05f82e2a9fc20b56af3` |
| #31 | merged | `main` | `codex/issue-009-plan` | `3b8f173501fea20fed3f35421886be503c7fdfec` |
| #32 | merged | `main` | `codex/issue-009-activate` | `447d070cdfd6b99d68c76e003f1a8803467d264d` |
| #33 | merged | GUI branch | `codex/fix-issue-9` | `9e77fcf628eb9caaf563da133e4186930769eb61` |
| #34 | merged | `main` | `codex/issue-015-plan` | `14df1b7fcea0775c56b6a774c4047c3f14e29c9d` |
| #35 | merged | `main` | `codex/issue-015-activate` | `13e18611b5db9cc4881b85a266d8259107641719` |
| #36 | merged | GUI branch | `codex/fix-issue-15` | `3ced61bcb9957907cc568e2e2560edd6f3c53b83` |
| #39 | merged | `main` | `codex/issue-037-plan` | `cba86f0e1f6cd2d03c6280ebda88e8714925af9a` |
| #40 | merged | `main` | `codex/issue-037-activate` | `ad46b0c878d99e1639f6bed7590162ab3b173d93` |
| #41 | merged | `main` | `codex/fix-issue-37` | `ef0d4d48b022b013e9dce75d9aaf62f0ddf3bf71` |
| #42 | merged | GUI branch | `codex/issue-038-evidence-contract` | `5e5e8941670f519e6447a770fe0f0d012b96cbcc` |
| #43 | merged | GUI branch | `codex/issue-038-closeout` | `90a541c00b3809a1190c2820e883266c40133f60` |

## Per-ref classification register

Every non-default remote head and every local-only head above is assigned to
exactly one family here. Family grouping avoids repeating the same inherited
entry points and defects; the immutable-tip tables remain the per-ref record.

### Current GUI lineage and issue forks

Refs: `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`,
`codex/fix-issue-6`, `codex/fix-issue-9`, `codex/fix-issue-15`,
`codex/issue-038-evidence-contract`, `codex/issue-038-closeout`, and
`codex/qwen-embedding-migration-plan`.

- Entry points: `rag_gui.py`, `streamlit_app.py`, `GUI_direct_search.py`,
  `Text_embeddings_to_chromadb_python.py`, and the file monitor, plus inherited
  CLI/email/provider scripts.
- Storage: direct Chroma and Milvus access with `html_chunks`,
  `html_chunks_temp`, and `html_chunks_text_in_documents`; no shared service
  boundary. Providers include Ollama, Groq, Kokoro/gTTS, and speech-recognition
  paths.
- Evidence: the current GUI tip has 19 test/fixture paths, but also 20 tracked
  bytecode files, two zero-byte logs, vendored browser assets, a Chroma SQLite
  file, and a `chroma-viewer` gitlink. Eight tracked locations contain personal
  path-shaped configuration or documentation.
- Salvage: only reviewed contracts already merged through PRs #29, #33, #36,
  #42, and #43; worker/progress, source/score, streaming, settings/theme, and
  reconnect concepts route to #20-#24/#27. The remote Qwen tip adds one
  proposal commit over inherited GUI-era work; the proposal is review material,
  not implementation or a merge authority.
- Disposition: preserve the current GUI tip for the proposed sanitized GUI
  archive; port tested contracts, never merge a whole issue/experiment fork.

### Legacy Streamlit, BM25, web, and QtPy lineage

Refs: `B25-hyb-kro-tkn-lmt-synms-mon-chngs-strmlit-chrdb-docs-web`,
`BM25-hyb-gtts-tkn-lmt-synms-mon-chngs-streamlit`,
`BM25-hyb-gtts-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`,
`BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`,
`streamlit`, `streamlit-mon-seq-align`, and
`codex/add-qtpy-gui-for-groq_lama_chromadb_rag_etts.py`.

- Entry points: `streamlit_app.py`, direct Groq/Ollama Chroma/Milvus scripts,
  ingestion/query/monitor utilities, and the PR #1 QtPy wrapper. The web tip
  additionally carries `web_search.py` and Kokoro trial code.
- Storage/providers: direct Chroma/Milvus with `html_chunks`,
  `html_chunks_temp`, and later `html_chunks_text_in_documents`; Groq, Ollama,
  gTTS/Kokoro and branch-specific web/model experiments are mixed into clients.
- Tests/artifacts: the older Streamlit/web/QtPy tips have no tests; the BM25
  Kokoro tip has six test paths. They carry 6-25 generated paths, including
  bytecode, `your_db_path/chroma.sqlite3`, vendored browser assets, cache files,
  and generated audio. The monitor tip's own commit says it is not working.
- Salvage: UI shape, streaming/search-only/source panes, file-change concepts,
  and PR #1's thin wrapper idea only. BM25-over-vector-candidates, direct
  provider/storage coupling, metadata assumptions, and broad fallbacks are
  known-broken and must not be ported.
- Concrete broken paths: the web helper has no success return and one caller
  supplies the wrong arity; the monitor tip records unresolved duplicate/
  embedding-function mismatches and writes metadata without the later document
  contract; the QtPy wrapper warms retrieval with a dummy query but sends user
  prompts directly to chat without per-prompt evidence retrieval.
- Disposition: archive only sanitized immutable tips needed to represent the
  divergent web and monitor histories; PR #1 is correctly closed/unmerged and
  its future destination is #24. Do not merge or rebase these refs.

### Narrow search, speech, and hotkey experiments

Refs: `hyb-gtts-tkn-lmt-synms-monitor-chngs`,
`hyb-gtts-tkn-lmt-synonyms`, `hybrid-coquiTTS`, `hybrid-gtts-tkn-lmt`,
`hybrid_search`, `media_keys`, and local-only `hybrid-gtts`.

- Surface: small direct-client experiments for hybrid search, synonym
  expansion, token limits, gTTS/Coqui speech, monitor launch actions, and media
  keys. They inherit the old direct Chroma/Milvus and Ollama/Groq scripts.
- Evidence: no maintained tests or service contract. `hybrid_search`,
  `hybrid-gtts-tkn-lmt`, `hyb-gtts-tkn-lmt-synms-monitor-chngs`, and local
  `hybrid-gtts` are already ancestors of the GUI tip; the other three are
  divergent.
- Concrete broken paths: synonym JSON is consumed as if it were already a
  mapping, MMR expects embeddings absent from constructed metadata, the legacy
  token counter can leave a zero response budget, Coqui loads a heavyweight
  model at import and reuses one temporary filename, and the media-key branch
  starts its listener twice with inconsistent pause/resume state.
- Salvage/disposition: convert only explicit provider-capability and UI hotkey
  requirements into #22/#24 tests. Preserve divergent Coqui/synonym/media-key
  tips until #26B decides sanitized archive versus verified deletion; no code
  merge.

### PubMed and cosine database experiments

Refs: `Pubmed_bert`, `cos_chroma`, local-only `Pubmed_Bert_cos`, and local-only
`simplifyDB`.

- Entry points: inherited ingestion and direct Groq/Ollama Chroma/Milvus
  clients. Collections include `Pubmed_HTML_chunks`,
  `Pubmed_cosine_HTML_chunks`, `cosine_HTML_chunks`, and `html_chunks`.
- Evidence: no tests. The history itself records the cosine path as not
  working. Tracked generated state includes `your_db_path/chroma.sqlite3`; the
  PubMed tip also contains an 18,103,674-byte `index_metadata.pickle` and a
  misspelled `Semantic_chuking` directory. Four tracked paths contain personal
  path-shaped configuration. PubMedBERT's 768-dimensional query path coexists
  with inherited Ollama `mxbai-embed-large` paths, so model/vector compatibility
  is unproven. `simplifyDB` has no unique commit versus `main`.
- Salvage/disposition: preserve model/collection provenance and potential
  read-only corpus evidence for #19/#20/#23; do not port the direct clients or
  generated database/index files. Keep the two unique PubMed/cosine tips until
  #19 export evidence and #26B disposition; `simplifyDB` is delete-after-
  verification once its recorded SHA is covered.

### Merged control-plane and local closure heads

Remote refs: `agent/canonical-migration-plans`, `codex/fix-issue-37`,
`codex/github-issue-plans`, `codex/issue-009-activate`,
`codex/issue-009-plan`, `codex/issue-015-activate`, `codex/issue-015-plan`,
`codex/issue-037-activate`, and `codex/issue-037-plan`. Local-only ref:
`codex/issue-015-close`.

- Eight remote tips are ancestors of current `main`; their source, tests, and
  canonical plans are already retained by `main`. The older
  `agent/canonical-migration-plans` tip is not a direct ancestor, but its evolved
  plan content was imported into the GUI lineage; compare canonical files
  before retiring it. Zero-path rows have no unique tree delta versus `main`.
- The local #15 closure tip changes only the canonical #15 plan/index and is
  already represented by the later GUI-lineage ledger state.
- Disposition: no separate archive. Retain exact SHAs through #26B evidence,
  then delete branch names only after merged-tree/ledger verification.

## Generated and private-path artifact register

| Class | Evidence | Disposition owner |
|---|---|---|
| Python bytecode/caches | Current GUI tip tracks 20 `*.pyc` files under root/test `__pycache__`; older tips track 3-16. | #26A removes tracking and verifies ignores. |
| Runtime logs/cache/audio | Zero-byte `error.log`/`output.log`; legacy `ffcache*`, `output.wav`, and `test_output.mp3`. | #26A deletes only after exact-tip preservation. |
| Local database/index state | `your_db_path/chroma.sqlite3` (155,648 bytes) on experimental tips; `index_metadata.pickle` (18,103,674 bytes) on `Pubmed_bert`. | #19 inventories/exports first; #26A must not destroy sole evidence. |
| Browser/viewer assets | `lib/vis-9.1.2/*`, other vendored browser files, and `chroma-viewer` gitlink bypassing the future backend. | #26A classifies license/source; #25/#26D retire unsupported viewer paths. |
| Personal paths | 4-8 tracked locations per experimental family contain user-profile/OneDrive-shaped literals; values are intentionally omitted here. | #26A replaces supported-path configuration; #26B handles history. |
| Historical credential findings | 12 Groq findings across six paths/two commits plus one generic finding in minified browser code. | #26B reconciles every reachable ref and Support handoff; never baseline away. |

## Proposed archive-ref register

These names are proposals only. #18 creates no tag, branch, or archive. #26B
must first sanitize/reconcile the recorded tip and then either create the
approved archive or record verified deletion coverage.

| Proposed archive ref | Recorded source tip | Covers |
|---|---:|---|
| `archive/easy-rag-main-before-postgres` | `f87b4cfcff4351c8f6e1e73ef43a7bf99e7fd5f9` | default/control-plane history |
| `archive/easy-rag-gui-experiments` | `f78d53476a452dd66aa207d74ebd7d2dca458fca` | merged GUI/BM25/fix/#38 ancestry |
| `archive/easy-rag-streamlit-web-experiment` | `355cf8560470b8752b5962da083d1599b9a9807c` | divergent web/model tip |
| `archive/easy-rag-streamlit-monitor-experiment` | `728123f102465843f46f87121e15d14d4008da0f` | divergent monitor tip |
| `archive/easy-rag-qtpy-pr1` | `8cb08091347ee41e7c8dac1c04d02a8e5d2eed2b` | PR #1 wrapper history |
| `archive/easy-rag-pubmed-experiments` | `3ceb6c2f4a04f4dd97e5e1bea674af1ee97408e6` | remote PubMed tip |
| `archive/easy-rag-pubmed-cosine-local` | `e0b2ac7c94589675dfbc8abebb3f7afec416f8dd` | divergent local cosine tip |

Narrow divergent speech/synonym/media-key refs remain explicit in the manifest;
#26B may add one sanitized archive per accepted requirement or record verified
deletion after #22/#24 disposition. An archive ref never substitutes for #19's
data export or #25's rollback evidence.

## `main`

- Tip at inventory snapshot: `f87b4cfcff4351c8f6e1e73ef43a7bf99e7fd5f9`
- Role: old upstream-derived CLI and email RAG code; this is the current default branch, not the advanced GUI implementation.
- Current problem: README and layout still describe the old Ollama/Chroma workflow and do not represent the PostgreSQL/pgvector target.
- Disposition: retain as the migration base. Add new PostgreSQL work in focused branches; do not replace it with an experimental branch merge.

## `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`

- Tip at inventory snapshot: `f78d53476a452dd66aa207d74ebd7d2dca458fca`
- Relation to `main`: verify live before execution; this dated snapshot is not
  a frozen implementation target.
- Main additions: several PyQt modules, `rag_gui.py`, `streamlit_app.py`, a large Chroma/Groq/Ollama application module, direct-search and context helpers, file monitoring, semantic chunking, Chroma utilities, TTS code, and GUI tests.
- Executable/client candidates: `rag_gui.py`, `streamlit_app.py`, `groq_lama_chromadb_RAG_ETTS.py`, the monitor batch file, and ingestion/query utility scripts.
- Database assumptions: direct ChromaDB access from clients and workers; multiple hard-coded or historical collection names, including `html_chunks_temp` and `html_chunks_text_in_documents`; no shared PostgreSQL service boundary.
- Provider assumptions: Ollama plus optional/embedded Groq paths, Google TTS and speech-recognition code. Local and cloud behaviour is mixed rather than expressed as explicit modes.
- Useful salvage:
  - background worker and progress/error patterns;
  - streaming result presentation;
  - search-only interaction;
  - source/context panes and score display;
  - database status/reconnect ideas;
  - deterministic chunk-ID and file-change concepts;
  - tests that can be converted into contract or regression fixtures.
- Known broken or unsafe areas:
  - legacy retrieval defects tracked by #5–#14;
  - BM25 is derived from vector candidates rather than an independent corpus search (#10);
  - collection access and provider logic are mixed into clients;
  - personal Windows paths and runtime downloads occur at import/startup;
  - broad exception handling hides retrieval failures;
  - generated `__pycache__`, `.pyc`, logs, caches and browser assets are committed.
- Disposition: **archive and mine selectively**. Do not merge wholesale. Port only behaviour that has a destination contract and tests under #20–#24 and #27.

## `codex/add-qtpy-gui-for-groq_lama_chromadb_rag_etts.py`

- Tip: `8cb08091347ee41e7c8dac1c04d02a8e5d2eed2b`
- Relation to `main`: 16 commits ahead, 0 behind.
- Relation to the current BM25 branch: diverged; 1 commit ahead and 6 commits behind, with merge base `b293135ff91befb0c27d9725b5e7001ff9cc1d0f`.
- Unique change: a 98-line QtPy wrapper plus one README line and one dependency line.
- Database/provider assumptions: inherits the older experimental Chroma/Groq application directly rather than using a shared search/chat boundary.
- Useful salvage: minimal Qt wrapper shape only; the future GUI requirements are already captured by #24.
- Disposition: **close PR #1 as superseded and archive the SHA**. Do not rebase or merge this branch.

## PR #1 — `Add QtPy GUI for Groq chat`

- State at inspection: **CLOSED, unmerged**.
- Base: the experimental BM25 branch, not `main`.
- Head: `8cb08091347ee41e7c8dac1c04d02a8e5d2eed2b` on the stale/diverged QtPy
  branch above.
- Decision: superseded by #17 and #24. Its useful idea is recorded as a future thin-client option; its architecture is not a merge path.

## PR #36 — Atomic vault write

- State at inspection: **MERGED** on 2026-07-24.
- Base: `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
- Reviewed code head: `3ced61bcb9957907cc568e2e2560edd6f3c53b83`.
- GitHub Create-a-merge-commit SHA:
  `88d0758ce1ffe6d61dd3ed99c0c5558e1bb8f205`.

## Closed fixes not present on `main`

Issues #3 and #4 were closed against commits `4686575a55358a0d5bf74447dc5e083e1361f66b` and `07b546964588cf9bce85ec35d9775a58b6266477`. Both commits are descendants of the experimental GUI history and are not part of `main` or either named branch tip inspected above.

Treat these commits as salvage evidence, not as proof that the default branch or future PostgreSQL client is fixed. If file opening or subprocess search is reintroduced, write new tests against the new service/client boundary and port the safety properties rather than cherry-picking the old application files.

## Archive and cleanup rules

1. Rotate/revoke exposed credentials before preserving any public archive ref.
2. Run a secret scan across all remote, tag, pull-request and local refs; publish only secret types/locations, never values.
3. Record each local-only branch name and tip SHA in this document before deletion.
4. #18 proposes archive refs only; #26B may create sanitized archives after its
   separately approved rewrite/reconciliation gate.
5. PR #1 is already closed/unmerged; retain its #24 disposition and exact SHA.
6. Delete no branch until #19 export evidence and #25 rollback gates exist.

## Closeout verification

- At checkpoint `6b87513`, the initial and closeout all-ref manifests differed
  only because the local working branch advanced from `7d1fd52` to `6b87513`.
  Remote heads, fetched PR refs, tags, and every other local head were unchanged.
- No `archive/*` local, remote, or tag ref exists. No GitHub ref or PR mutation
  was performed.
- A second all-history scan still found the same 13 findings after sorting the
  redacted detector/commit/path/fingerprint records; remediation remains #26B's
  gate.
- The ledger validator, required module compilation, and 145 non-GUI tests pass
  at `6fd0516`. The full suite cannot collect the two PyQt GUI modules in this
  environment because `PyQt5` is not installed.
