# Issue #11 Chroma embedding contract implementation plan

**Status — completed.** [PR #57](https://github.com/sriharshaguthikonda/easy-local-rag/pull/57) activated #11. [PR #58](https://github.com/sriharshaguthikonda/easy-local-rag/pull/58) implemented and independently verified the contract using synthetic Chroma. [Closure evidence](https://github.com/sriharshaguthikonda/easy-local-rag/issues/11#issuecomment-7200752206).

Frozen source baseline for implementation is maintained GUI branch
`GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` at
`32f81f748f98e2009e1de97c6c140df37405fd94`. Recheck the live tip at activation. If it has moved, update and re-review the frozen base before code work. Keep the approved branch lineage independent of main. The scope is the maintained local Chroma path
defined by ADR 0003. Tests use Python, pytest, temporary Chroma stores, and
mocked Ollama. [Issue #11](https://github.com/sriharshaguthikonda/easy-local-rag/issues/11) ·
[Acceptance contract](../../issues/ISSUE-011-embedding-model-match.md) ·
[Maintained entry points at the source baseline](https://github.com/sriharshaguthikonda/easy-local-rag/blob/32f81f748f98e2009e1de97c6c140df37405fd94/docs/decisions/0003-maintained-entry-points.md) ·
[Roadmap ledger](../../issues/README.md)

## Approval and boundaries

The [owner approved an independent #11-only exception](https://github.com/sriharshaguthikonda/easy-local-rag/issues/11#issuecomment-7191003443).
Merge the canonical documentation and then a separate PR declaring #11 the
sole active issue. #2 closure, #19/#20, #10, live collection data and metadata,
and PostgreSQL migration are outside this packet. #10 remains blocked until
#11 has passed independent verification.

Keep `embedding_model` as the current exact, case-sensitive model identifier.
Preserve labels on existing collections and keep the explicit offline lexical
path callable with unknown provenance. Use read-only lookup for readers. Any
re-index or adoption of an existing unlabelled collection is a separate owner
decision. This packet leaves NPU and model-revision migration outside scope.

## Maintained caller map

The reporters and launchers below keep their public result shapes. Shared
assertions use synthetic providers and inspect both provider and Chroma calls.

| Route and owner | Guard and regression |
| --- | --- |
| `GUI_direct_search.DirectSearchSubprocessWorker.run` and `DirectSearchMixin._query_via_subprocess` | Guard phrase, hybrid and self-test before embed/query. Cover in `test_direct_gui_checks_selected_model`. |
| `rag_gui.RAGChatGUI.get_relevant_context_hybrid` → `GUI_context.get_relevant_context_hybrid` → `GUI_chromadb_helper.main/query_chromadb` | Launcher delegates without embedding. Helper opens, validates, embeds and queries. Cover in `test_gui_chat_and_helper_reject_incompatible_model`. |
| `GUI_workers.ChromaDBSearchWorker.run` | Same guard before the standalone worker's embedding. Cover in `test_gui_search_worker_checks_provenance`. |
| `streamlit_app.get_relevant_context_hybrid` → `streamlit_groq_lama_chromadb_RAG_ETTS.get_relevant_context_hybrid` | App delegates. Backend checks before embed/query, including same-dimension mismatch. Cover in `test_maintained_backend_checks_before_embedding`. |
| `groq_lama_chromadb_RAG_ETTS.main/get_relevant_context_hybrid` and `context_retrieval.get_relevant_context_hybrid` | ADR 0003 lists the first script as maintained, including its CLI. Guard both retrieval paths and eliminate create-on-read. Cover in `test_maintained_backend_checks_before_embedding`. |
| `monitor_file_changes_update_chromaDB` startup and `FileChangeHandler.process_file` | Guard collection initialization and the per-file embedding/write. Cover in `test_watcher_leaves_unlabelled_collection_intact`. |
| `Text_embeddings_to_chromadb_python.main/init_chromadb/insert_batches_to_chromadb` | Validate the artifact and destination before touching the database. Cover in `test_csv_preflight_precedes_destination_writes`. |
| `GUI_settings.py`, `GUI_chromadb.py` and `offline_search.py` | Settings and collection selection supply the request, while explicit lexical lookup makes no vector calls. Cover the setting change in the direct GUI regression and lexical behavior in `tests/test_offline_search.py`. |

## 1. Reproduce the defects before any product edits

- [x] Extend `tests/test_embedding_contract.py` with `test_unknown_count_error_rejects_read_and_write`, `test_existing_collection_metadata_is_never_modified`, and `test_writer_create_race_revalidates_collection`. Assert zero `modify` and `add` calls when unknown provenance is denied. Run fresh/matching/empty tests alongside wrong-model and unreadable-count fixtures.
- [x] Add `tests/test_chroma_embedding_entrypoints.py`. Drive the actual GUI direct phrase/hybrid subprocess, GUI chat/helper, GUI self-test and search worker, Streamlit backend, maintained CLI, and `context_retrieval.py`. Change the model via GUI settings. For missing non-empty provenance, unreadable count, same-dimension mismatch and different-dimension mismatch, assert zero embedding and Chroma query calls. Verify a prelabelled match reaches both, a verified empty reader yields no results, and a missing collection is never created by retrieval. Mark guards already correct as positive controls.
- [x] In that file exercise the watcher and CSV entry point. Reproduce both writers stamping and writing when count raises. Reproduce CSV accepting unlabelled vectors, then add manifest mismatch, digest mismatch, wrong vector shape, non-finite values in a later batch, and matching provenance cases. Invalid input must precede any collection creation or `add`. Include a real temporary Chroma fixture proving equal dimensions permit a mismatched vector on the current Streamlit path. Confirm the existing offline lexical path still returns text and IDs while vector retrieval is refused.
- [x] Run `python -m pytest tests/test_embedding_contract.py tests/test_chroma_embedding_entrypoints.py tests/test_offline_search.py -q` on the unfixed source. Capture the failure per confirmed defect and matching-case passes. Keep the tests in the first, RED commit only after activation. Use injected clients or subprocess fixtures so imports cannot open configured stores or use network connections.

## 2. Make collection validation fail closed

- [x] In `embedding_contract.py`, add `validate_embedding_compatibility(collection, requested_model, *, for_write=False) -> bool`. Return `True` for an exact valid named match and `False` only for a read of an unlabelled collection with a verified zero count. Raise a typed contract error for missing or malformed provenance, a mismatch, or an attempted count that raises or returns a boolean, non-integer or negative value. A known label needs no count to establish model compatibility. Readers branch on `False` and return the public empty result before provider/query calls. Writers handle contract errors before work and never treat them as empty.
- [x] Change `monitor_file_changes_update_chromaDB.py` and `Text_embeddings_to_chromadb_python.py` to use a shared `open_collection_for_embedding_write(client, name, model)` in `embedding_contract.py`. Open an existing collection and require an exact label. On `NotFoundError`, call `create_collection` with initial `embedding_model` and existing creation options. On a duplicate, reopen and validate the winner. Cover both `UniqueConstraintError` and the installed Rust backend's `InternalError` with the exact `Collection [<name>] already exists` message. Propagate unrelated errors. Eliminate `modify` from both maintained writer paths. A pre-existing unlabelled empty collection returns an explicit switch/create error for writes. A newly created collection is labelled at creation and can accept its first write.
- [x] Recheck the exact model on the same opened collection before ingestion work. No write or watcher embedding on validation failure. Keep corpus verification limited to temporary fixtures.

## 3. Apply the contract at every maintained vector boundary

- [x] Use the helper in `GUI_direct_search.py` for both modes and self-test, and in `GUI_workers.py` before the search worker embeds. Preserve the captured `GUI_settings.py` model value through validation and embedding. In `GUI_context.py` and `GUI_chromadb_helper.py`, send query text and the selected model to the existing subprocess. Open, validate, embed, and query the same collection in that subprocess. Replace the raw-vector subprocess request so it cannot bypass the check.
- [x] Validate before embedding in `streamlit_groq_lama_chromadb_RAG_ETTS.py`, `groq_lama_chromadb_RAG_ETTS.py`, and `context_retrieval.py`. Replace retrieval-side `get_or_create_collection` with `get_collection` wherever used. Keep the Streamlit evidence return shape and visible failures. Surface a concise model mismatch, unlabelled collection, or unavailable-collection error, with no embedded vectors logged. Verified-empty readers return their normal empty result. Leave `offline_search.py` independent of vector validation.

## 4. Require provenance at CSV import

- [x] In `Text_embeddings_to_chromadb_python.py`, require a producer-supplied `<file>.csv.provenance.json` with `schema_version` `1`, exact `embedding_model`, positive integer `dimensions`, and lowercase `csv_sha256`. Compare the declared model with the configured writer and destination. Copy in 1 MiB blocks into a private temporary directory. Use the captured source size as the copy cap and reject changed size or I/O failures. Validate the digest and every row from that copy, then consume the same copy in bounded batches. Clean up the temporary directory on success and failure. Open or create Chroma only after preflight and explicitly require the stored label to match the manifest before the first write. Existing partial-write semantics for unrelated backend failures remain visible. Validate the current five CSV fields and require a nonempty, finite numeric JSON vector of the declared length on each row. Reject missing, malformed, mismatched or changed artifacts before initializing the destination. The importer does not infer provenance from vector shape or create a sidecar itself.
- [x] Test a valid existing destination, a genuinely new collection, a competing create with a different model, a stored destination mismatch despite a valid manifest, both duplicate exception forms, and a bad later batch with zero writes. Keep existing chunk IDs and batch write behavior for validated input.

## Acceptance and rollback

- [x] Run the focused command from step 1, then `python -m pytest tests -q` and `python -m py_compile embedding_contract.py GUI_direct_search.py GUI_context.py GUI_chromadb_helper.py GUI_workers.py streamlit_groq_lama_chromadb_RAG_ETTS.py groq_lama_chromadb_RAG_ETTS.py context_retrieval.py monitor_file_changes_update_chromaDB.py Text_embeddings_to_chromadb_python.py`. Record RED and GREEN results, every exercised entry point, and mutation-spy assertions in the review evidence.
- [x] After activation, keep the RED test commit and GREEN implementation commits on one topic branch from the frozen GUI tip. Review on the exact head with one independent reviewer, run both the focused and full suite from an independently provisioned synthetic fixture, and merge the verified head to the GUI branch. Record the commit, commands and immutable evidence on #11, then synchronize the canonical ledger as a follow-up. A moved base requires exact-head re-verification. Existing labels and data need no rollback. #10 hybrid work consumes the verified guard as a separate packet.

### Rollback boundary

If a post-merge defect requires reversal, the code owner opens a reviewed PR
reverting only the #11 product merge on the maintained GUI lineage. Revert
validation uses a synthetic store. Deploying that old revision to the real
store is a separate operator action. Keep the verified guarded checkout for
running jobs and use explicit lexical-only search during the rollback window.

Before selecting the reverted checkout on Windows, the operator exits the Qt
and Streamlit apps, stops the maintained CLI and watcher, and suspends any
configured launcher for the watcher or CSV import. Stop any service importing
`context_retrieval.py`. Verify the known entry points are stopped from a
PowerShell prompt in a separate session:

```powershell
$pattern = 'rag_gui[.]py|streamlit_app[.]py|groq_lama_chromadb_RAG_ETTS[.]py|monitor_file_changes_update_chromaDB[.]py|Text_embeddings_to_chromadb_python[.]py|context_retrieval[.]py'
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -match $pattern
}
if ($running) { throw 'A maintained vector or ingestion entry point is still running.' }
python -m pytest tests/test_offline_search.py -q
```

The offline test uses temporary fixtures. Confirm service/scheduled launchers
are suspended as well. The reverted vector and ingest modes stay out of
service until a corrected, independently verified guard is available.
