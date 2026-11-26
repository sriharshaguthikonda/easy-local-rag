# Issue #10 Chroma hybrid retrieval implementation packet

**State** Closed after implementation, verification, and issue closure.
**Mode** Maintained ChromaDB.
**Target** `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
**Frozen base** `0be9ebe688ffd795960b6761422087105d26e63d`.
**Authority** [Issue #10 approval](https://github.com/sriharshaguthikonda/easy-local-rag/issues/10#issuecomment-7200914319) · [Canonical acceptance](../../issues/ISSUE-010-full-corpus-bm25.md) · [Ledger](../../issues/README.md) · [Prior art](../../Research/landscape.yaml).

This packet incorporates the owner-provided Chroma hybrid retrieval draft. The independently verified #11 merge supplies the embedding contract. The approval applies to #10 while #2 is open. Implementation begins after a separate activation merge. All tests and performance probes use temporary synthetic collections. Credentials, live data, migration and the #2 closure gate are outside scope.

## Locked interfaces and behavior

- Return ranked evidence records with `collection_id` (Chroma UUID string), `chunk_id` (exact Chroma record ID), `document`, `meta`, `distance` (number or null) and `final_score`. Join by the pair of IDs. File name stays display metadata. Preserve fields and ordering through GUI and generation source adapters, including Streamlit's numbered citations. Unique source groups use full path where present and retain all selected chunk IDs.
- Use existing Chroma `query` for vector retrieval and Python's SQLite FTS5 `bm25()` for lexical retrieval, with the `unicode61` tokenizer and OR of quoted original-query terms. Use `where` on both legs. For the GUI's case-insensitive filename/snippet substring, collect eligible IDs per page and use Chroma's supported `query(ids=...)` to merge bounded vector lists before final scoring. Return empty when the filter matches zero records.
- For each enabled component, order by its native score then stable identity. Weighted RRF adds `weight / (60 + rank)` with one-based ranks from each leg. Weight zero omits that leg's contribution. A duplicate ID counts once per leg. Retain vector distance and embeddings when present, including when the same ID matches lexically. Secondary keyword/synonym and phrase controls score the captured candidate union where the caller already supports them. Keep each caller's output shape, ordinary phrase mode, and explicitly selected offline lexical mode. MMR diversity uses vector embeddings when available and keeps the fused top K eligible for display and citation.
- Open an isolated native client in a fresh operation process for managed persisted reads. The initial guarded preflight performs only collection lookup, metadata and count, so no HNSW index is opened before capture. GUI search already has a subprocess. Use the shared request worker for the maintained chat/backend adapters. Validate #11 before embedding, perform the provider call outside the store lock, then reacquire the same store lock, look up a fresh collection object by name and recheck its UUID and model immediately before capture. This lock covers every vector query and lexical page read. If either identity or compatibility changed during embedding, fail closed before querying. A same-UUID content update before capture is read in its entirety by the fresh operation process. Return the normal empty output for a verified empty collection. Lexical-only reads invoke no provider and never stamp embedding metadata.
- Coordinate reader capture, watcher collection creation/add and CSV creation/import using the same exclusive advisory lock for the canonical absolute store path. Put lock files in the current account's application state area. Use OS-held locks on Windows and POSIX, a 15-second acquisition deadline, and automatic release on process exit. The watcher reopens a fresh native client for mutation and checks current IDs and model before adding. Keep CSV source attestation before the first destination write and group its batches under one lock. Direct writes through other tools are outside the coordinated runtime.
- Capture up to 512 Chroma rows per page. Build a private request-scoped FTS5 index on disk with a 4 MiB SQLite page cache and disk-backed temporary storage. There is no cross-request index reuse. Additions, deletes, same-count text/metadata edits and collection switches are visible on the next query. Use SQLite max_page_count to enforce a 2 GiB main scratch database and disable journaling for that disposable file. Check the 512 KiB UTF-8 text per chunk before indexing and enforce a 32 MiB aggregate UTF-8 evidence budget across the retained candidates. The page itself is upstream of the per-chunk size check. Bound a capture to 90 seconds between native calls. Candidate windows are `max(50, top_k + additional)`, capped at 1,000 per leg. Report an explicit error on timeout, malformed/duplicate IDs, missing FTS5, or size limits. Do not return a partial ranking. Scratch state is cleaned on normal completion and exceptions. The supported reader boundary accepts the store, selected collection, query and filters without any writer capability or generation credentials.
- GUI distance sorting puts missing or non-finite distances last for either direction. Send the active filter with the search request and only display a completion belonging to the current worker, store, collection and filter. Changing the filter invalidates the displayed request until Search is run again.

## Execution and evidence

1. **RED tests only.** Replace source assertions in `tests/test_gui_direct_search_bm25.py` and add behavioral fixtures in `tests/test_chroma_hybrid_retrieval.py`. Drive the existing GUI subprocess and maintained hybrid entry points. Reproduce basename collision, lexical-only hit outside 50 vector candidates, mixed distances in both sort directions, pre-limit filter behavior, stale GUI completion and stale native state across a process update. Keep positive controls for behaviors already correct. Record the defect outputs and commit only tests.
2. **Identity, fusion, filtering.** Own `chroma_hybrid_retrieval.py`, `GUI_direct_search.py` and the direct-search connections in `rag_gui.py`. Use explicit identity and compare exact order in fixtures, including two collections that share a chunk ID, equal distances, duplicate leg inputs, zero lexical weight, original query terms and provider-independent lexical retrieval. Verify text and IDs survive citation construction.
3. **Candidate and adapter work.** Update `GUI_context.py`, `GUI_chromadb_helper.py`, `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `groq_lama_chromadb_RAG_ETTS.py`, `context_retrieval.py` and `offline_search.py` as required by their entry points. Keep the Streamlit `(context, sources)` contract and the CLI/context evidence shape. Test returned lexical-only and vector-only evidence for each adapter. Preserve MMR and original-source formatting.
4. **Consistency.** Own the store-lock boundary in `chroma_hybrid_retrieval.py`, `monitor_file_changes_update_chromaDB.py` and `Text_embeddings_to_chromadb_python.py`. Assert readers are blocked during a controlled multi-step mutation using a separate process and observe the complete new revision, including swapped vector order at constant count. The committed writer updates A and B in two steps inside its lock. Capture must wait and then return B's new text/vector rank and current metadata under B's original ID. Separately replace the collection (new UUID) or change its embedding model between the preflight and capture, and assert that no vector query runs. Test addition, deletion, same-count text/metadata update, same-name recreation, independent paths, where predicates and stale GUI completion. Assert a malformed or over-budget capture gives an error and using the same query afterwards opens a clean index. Measure page sizes, elapsed time and Python/SQLite memory on a synthetic corpus with more than 25,000 rows.
5. **Focused and full verification.** Run the commands below under provider/network denial with synthetic fixtures. Run the full suite and compile touched Python. An independent reviewer inspects the exact implementation head. A separate clean checkout runs verification before merge. A moved target requires re-verification. Publish the reviewed and merged SHAs, RED receipt, focused and full results, and concurrency/size evidence on #10. Close #10 only after the implementation merge and then reconcile the canonical ledger.

```powershell
python -m pytest tests/test_gui_direct_search_bm25.py tests/test_chroma_hybrid_retrieval.py tests/test_offline_search.py -q
python -m pytest tests/test_embedding_contract.py tests/test_chroma_embedding_entrypoints.py tests/test_streamlit_hybrid_embeddings.py tests/test_offline_search.py -q
python -m pytest tests -q
python -m py_compile chroma_hybrid_retrieval.py GUI_direct_search.py GUI_chromadb_helper.py GUI_context.py rag_gui.py offline_search.py context_retrieval.py groq_lama_chromadb_RAG_ETTS.py streamlit_groq_lama_chromadb_RAG_ETTS.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py
```

## Rollback

Revert the #10 merge on the maintained GUI lineage in a reviewed PR, then verify the embedding guard and explicit lexical mode on temporary collections. No collection or evidence IDs are rewritten by this change. Existing operator processes and the live corpus are outside this implementation and test run.

## Implementation evidence

The [locked packet PR #60](https://github.com/sriharshaguthikonda/easy-local-rag/pull/60)
and [activation PR #61](https://github.com/sriharshaguthikonda/easy-local-rag/pull/61)
merged before the tests-only RED commit `393309b` reproduced nine defects.
[Implementation PR #62](https://github.com/sriharshaguthikonda/easy-local-rag/pull/62)
merged as `89514bf502a39776b2261ff1657146f530364765`. Its tree matches the
independently reviewed head `55a915433e44f5cf0be66453d57a4e9670784021`.

The clean-checkout run with provider access denied passed 344 tests. The
focused #10 repeat passed 13 tests, including 11 isolated native Chroma
scenarios. The focused #11 run passed 90 tests. Touched Python compiled.
The separate writer and native watcher probes verified consistent managed
ingestion and retrieval. [Closure evidence](https://github.com/sriharshaguthikonda/easy-local-rag/issues/10#issuecomment-7201734064)

The lexical benchmark captured 25,601 synthetic rows in 51 pages with at
most 512 rows per page. It used a 4 MiB SQLite cache and measured a 1.49 MiB
traced Python peak and a 13.81 MiB scratch SQLite file. The figures describe
the measured lexical scan.

Chroma 1.0.16's native filtered ANN could omit eligible IDs in the synthetic
fixture, including with explicit `ids`. To keep the filter eligibility
guarantee, the filtered vector leg ranks bounded pages of eligible stored
embeddings using the collection's configured distance. The unfiltered leg
uses Chroma's native vector query. The #11 guard is applied before embedding
and checked again when the collection is reopened for capture.
