# Issue #15 Atomic Vault Write Implementation Packet

**Goal:** make the maintained HTML vault builder idempotent and make every `vault.json` replacement atomic without losing the previous valid target on write failure.

[GitHub Issue #15](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15) · [canonical plan](../../issues/ISSUE-015-atomic-vault-write.md) · [roadmap ledger](../../issues/README.md)

**Docs packet lineage:** `codex/issue-015-plan`, based on `origin/main` at `07fdddad6d2377ed3d147f7ba0de8a65c83ebdff`. Git history carries the packet commits; the immutable reviewed docs-merge SHA is required before the separate activation PR may be opened and merged.

**Code lineage:** create the implementation branch directly at frozen target head `619365224f6db3770d3369fd84a295589699e513`, target branch `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`. That SHA is the merge of [PR #33](https://github.com/sriharshaguthikonda/easy-local-rag/pull/33) and is also the exact `origin/GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` head used for this packet. Never merge or rebase main/docs into the code lineage.

**Mode:** maintained path. `Vault_json_creation_from_HTMLs.py` remains runnable. Issue #8 is still open: #15 may retain the already-landed `EASY_RAG_VAULT_SOURCE_DIR` override with the existing folder-picker fallback, but it must not claim #8 closure or edit #8-owned configuration/docs. No dependency change is allowed. The only schema evolution is the canonical gate's additive `content_hash`; legacy entries without it remain readable and are upgraded when their source is next processed.

**Worker-owned code/test files:** `vault_store.py`, `Vault_json_creation_from_HTMLs.py`, `tests/test_vault_store.py`. `.gitignore` is verification-only because `*.json` and `vault.json` are already ignored. Do not modify another source, test, configuration, dependency, generated vault, or documentation file.

## Authority and source coverage

GitHub remains authoritative for live state and owner decisions; the canonical plan fixes scope and closure gates; the roadmap fixes sequencing/status; this packet locks implementation detail without weakening either source.

| Source | Required item | Packet coverage |
|---|---|---|
| GitHub issue and owner plan comment | Read existing list, normalized `file_name` identity, unchanged keep, changed replace, corruption backup, same-directory temp/replace, no append | Locked below in schema, merge, failure, and tests |
| [Owner two-commit decision](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15#issuecomment-5070864427) | Exactly two named TDD commits, separate per-finding review-fix commits, no squash before review, and no activation before a separate activation PR merges | Commit names and lifecycle are locked below without changing any other gate |
| Canonical #15 plan | Maintained/retirement choice, stable identity/version semantics, atomic cleanup, four named regressions, rollback | Maintained mode and every named regression locked below |
| Roadmap ledger | #9 closes before #15; #8 fallback does not close #8; one active issue maximum | #9 evidence frozen; #15 remains planning with no active slot until the separate activation PR merges |
| Frozen GUI code/tests | Existing helper signatures, builder wiring, current four-test baseline, existing #8 env/picker fallback | Caller map, ownership, TDD baseline, and exact changes locked below |

There is no phase `RESEARCH.md`, `CONTEXT.md`, or requirement-ID artifact for this legacy issue packet. Live code inspection at the frozen SHA supplies implementation evidence; no source item is omitted or deferred by this packet.

## Frozen caller and entry-point map

Derived from the frozen SHA with:

```powershell
git grep -n -E "load_vault|merge_vault_entries|atomic_write_json|convert_html_to_json|Vault_json_creation_from_HTMLs|vault\.json" 619365224f6db3770d3369fd84a295589699e513 -- '*.py' '*.md'
```

- `vault_store.py:9`, `:33`, and `:58` define the three public storage helpers. Their only production importer is `Vault_json_creation_from_HTMLs.py:12`; their only other importer is `tests/test_vault_store.py:4`.
- `Vault_json_creation_from_HTMLs.py:91` defines `convert_html_to_json`; it loads at `:93`, merges at `:133`, atomically writes at `:134`, and invokes `main()` only under the `__main__` guard at `:153-154`.
- No Python module imports or calls `convert_html_to_json`. Its maintained entry points are direct script execution, `main()`, and explicit import/call.
- `Ollama_RAG_chat backup.py`, `Ollama_RAG_chat.py`, and `Ollama_RAG_chat_TTS.py` define unrelated `load_vault_content` readers. `Generate_embeddings.py`, `PubMedBert_generate_embeddings.py`, `Delete_processed_chunks.py`, the monitor, and semantic-vault builders independently read/write vault-like files. They do not call `vault_store` and are outside #15 ownership.
- Frozen `tests/test_vault_store.py` has four passing tests: missing load, mtime replacement, valid atomic output, and invalid-JSON backup. It has no unchanged-repeat, content-hash version, cleanup/failure-preservation, builder integration, import-side-effect, or append-mode regression.

## Locked public and private interfaces

Keep these public names; do not add a storage class, adapter, dependency, or second writer:

```text
load_vault(path: str | os.PathLike[str]) -> list[dict[str, Any]]
merge_vault_entries(existing_entries: list[dict[str, Any]], new_entries: list[dict[str, Any]]) -> list[dict[str, Any]]
atomic_write_json(path: str | os.PathLike[str], data: list[dict[str, Any]]) -> None
convert_html_to_json(directory_path: str | os.PathLike[str], vault_path: str | os.PathLike[str] = "vault.json") -> None
```

`vault_store.py` owns exactly these private interfaces; do not add another helper:

```text
_validate_vault_data(data: object) -> None
_normalized_file_identity(file_name: str) -> str
_invalid_backup_path(vault_path: Path) -> Path
```

`_validate_vault_data` is non-mutating, returns only `None`, and raises `ValueError` for invalid in-memory data. `_normalized_file_identity` is non-mutating, returns the normalized identity string, and raises `ValueError` for an empty/non-string name. `_invalid_backup_path` is non-mutating, returns the first currently unused sibling `Path`, and does not create, move, or reserve it. The three public storage helpers and `convert_html_to_json` return exactly the values declared above; `load_vault` returns newly decoded objects, `merge_vault_entries` always returns a new outer list, and both writers return `None`. A same-version merge list contains the original existing entry object; a replacement/hash upgrade contains a new shallow entry dictionary whose nested winning values are reused, not mutated or deep-copied. No helper mutates caller-owned lists, dictionaries, chunks, or paths.

Private names may not become public exports. `Vault_json_creation_from_HTMLs.py` owns `_get_sentence_tokenizer() -> Any` for lazy tokenizer loading; existing `clean_text`, `extract_text_from_html`, `generate_chunk_id`, `split_into_chunks`, `convert_html_to_json`, and `main` remain public by compatibility.

`path`/`vault_path` accepts strings and `os.PathLike`. Relative output paths resolve from the current working directory. The builder expands `~`, resolves the source directory to an absolute path without requiring #8 changes, raises `FileNotFoundError` for a missing source and `NotADirectoryError` for a non-directory, and persists each `file_name` as `os.path.normpath(str(Path(file_path).resolve()))`. Storage identity is `os.path.normcase(os.path.abspath(os.path.normpath(file_name)))`; therefore Windows case/separator aliases identify one file, while moving a file creates a new identity. Empty/non-string identities are invalid data.

## Locked vault schema, merge order, and limits

The on-disk document is exactly one JSON array. Every entry is an object requiring:

- non-empty string `file_name`, persisted by the builder as the normalized absolute path;
- finite numeric `modification_time`, excluding `bool`;
- list `chunks`; every persisted chunk is an object with lowercase 64-hex SHA-256 string `id` and string `text`;
- optional lowercase 64-hex SHA-256 string `content_hash`; the builder always emits it as SHA-256 of the UTF-8 cleaned full document text.

Unknown entry/chunk keys with string names and JSON-serializable values are preserved for compatibility but are not interpreted. All nested object keys must be strings and all numeric values at any depth must be finite; `json.dumps(value, allow_nan=False)` must accept every unknown value. A missing `content_hash` is valid legacy data. There are no new issue-specific byte, entry, or chunk limits: this local utility continues to load and serialize the array in memory, bounded by Python memory and available same-volume disk. A root other than a list, a non-object entry/chunk, a missing/invalid required field, an invalid optional hash, malformed UTF-8, or malformed JSON is corrupt input.

Merge identity never contains mtime, chunk IDs, or hashes. Existing duplicate identities collapse deterministically: the last existing value wins while retaining the identity's first position. Incoming duplicates behave the same. New identities append in incoming first-seen order. For an existing identity:

- same mtime and same hash returns the existing entry object/value unchanged;
- same mtime with both hashes absent returns the legacy existing entry object/value unchanged;
- an incoming hash upgrades an existing missing hash;
- a changed mtime or changed non-null hash replaces the entry in its existing position;
- an incoming missing hash does not downgrade an existing hash when mtime is unchanged.

Replacement and hash upgrade construct a new entry without mutating either input: incoming `file_name`, `modification_time`, `chunks`, and `content_hash` (when present) are canonical; existing-only unknown entry keys are copied forward; incoming unknown-key collisions overwrite existing unknown values. On a changed-mtime replacement, omission of incoming `content_hash` does not copy the stale existing canonical hash. A same-version keep is byte/object-value stable and returns the existing entry object at the same position. Tests deep-copy both inputs and assert equality after every merge, assert object identity/value on keep, and assert canonical/chunk/unknown-key winners on replacement and hash upgrade.

The builder processes each discovered HTML-family file, computes `content_hash`, and lets the merge helper decide unchanged versus replacement. One unchanged rerun remains one entry; changed cleaned content with an unchanged filesystem mtime replaces exactly once.

## Locked corruption, serialization, and failure contract

`load_vault` returns `[]` only for a missing or zero-byte target. Valid schema returns newly decoded data. It calls `_validate_vault_data` after decoding. `UnicodeDecodeError`, `json.JSONDecodeError`, or schema `ValueError` means corrupt on-disk input; other read/stat permission or I/O failures propagate and are not mislabeled as corruption.

Corrupt input is moved before recovery to the first unused sibling named `vault.json.invalid.<UTC-YYYYMMDDTHHMMSSffffffZ>.<pid>[.<counter>].bak`; UTC text is exactly `datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")` and pid is `os.getpid()`. The unsuffixed candidate is first and collision counters start at `.1`. `_invalid_backup_path` checks candidates deterministically and the move primitive is exactly `os.rename(vault_path, backup_path)`, never `os.replace`, so the implementation does not intentionally overwrite an existing backup. This packet does not add a concurrent-reader/writer reservation protocol. The backup preserves corrupt bytes exactly, `load_vault` prints one warning naming it, and returns `[]`. Backup-move failure propagates the original exception and leaves the corrupt target byte-for-byte unchanged.

`atomic_write_json` first calls `_validate_vault_data(data)` without mutating `data`. Only after validation succeeds does it always call `vault_path.parent.mkdir(parents=True, exist_ok=True)`, before temp creation; invalid data therefore cannot create a missing parent. It snapshots an existing target's permission bits with `stat.S_IMODE(vault_path.stat().st_mode)`. It then creates one sibling temp through `tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=f".{vault_path.name}.", suffix=".tmp", dir=vault_path.parent, delete=False)`. `handle.name` is the cleanup path; the temp is in exactly `vault_path.parent`, so Windows `os.replace` remains same-volume.

Serialization is deterministic: one `json.dump(data, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)`, followed by one explicit `handle.write("\n")`. The observable success order is exactly `dump -> flush -> os.fsync(handle.fileno()) -> close -> os.chmod(temp_path, saved_mode)` only when the target existed -> `os.replace(temp_path, vault_path)`. A missing target retains the platform's secure tempfile mode and skips chmod. `os.replace` is called exactly once and only after the handle is closed, which is required for Windows. Directory fsync is not added: the maintained target is Windows, and the canonical gate requires atomic visibility through same-volume `os.replace`, not a new cross-platform power-loss durability contract.

No append open and no second write branch may exist. On JSON/encode/flush/fsync/close/chmod/replace failure, the caught primary exception remains authoritative and the implementation attempts to close any still-open handle and unlink the named temp if it exists; replacement has not completed, so the old target remains byte-for-byte unchanged. If close or unlink cleanup fails, that cleanup exception never replaces the primary exception. The implementation uses `BaseException.add_note` to record the residual temp path and cleanup failure on the primary exception, then re-raises that same primary exception object. A residual temp may physically remain when unlink itself fails; the cleanup-failure test requires and records that residual instead of claiming successful deletion.

`Vault_json_creation_from_HTMLs.py` has no import-time download, tokenizer load, Tk window, pool, file scan, or vault write. `_get_sentence_tokenizer()` first loads local Punkt; only a runtime `LookupError` may trigger the existing quiet Punkt download and retry. `main()` alone applies #8's current rule: use an existing `EASY_RAG_VAULT_SOURCE_DIR`, otherwise open the existing picker; cancel means return without writing. `convert_html_to_json()` never opens UI and propagates storage/source/tokenizer errors with a nonzero CLI exit.

## TDD test matrix

Replace `tests/test_vault_store.py` first. It contains exactly these 17 named test functions, with no parametrization that changes the reported count. Tests use only `tmp_path`, `monkeypatch`, stdlib, and installed project dependencies:

| Test ID | Exact proof |
|---|---|
| `test_load_missing_and_zero_byte_return_empty` | Missing and zero-byte paths both return `[]`; neither creates a backup. |
| `test_unchanged_rerun_keeps_one_entry_and_object_value` | Merge then write/read twice with one normalized identity and identical mtime/hash; result length is one, the merge returns the existing entry object at its original position, its object value and deterministic serialized bytes are unchanged, and both input lists still equal deep copies. |
| `test_changed_mtime_with_unchanged_or_absent_hash_replaces_once` | Two independent merges use one stable normalized identity represented with a native `.`/`..` alias and changed `modification_time`: one keeps the same hash and one has hashes absent on both versions. Each remains one entry; incoming canonical fields/chunks win exactly once at the original position. This is independent of hash-change behavior. |
| `test_same_mtime_changed_hash_replaces_once` | One stable normalized identity with unchanged mtime and different 64-hex `content_hash` remains one entry; incoming canonical fields/chunks/hash win exactly once at the original position. This is independent of mtime-change behavior. |
| `test_legacy_hash_upgrade_merge_winners_and_no_mutation` | A same-mtime legacy entry missing hash upgrades from an incoming hash; incoming canonical fields/chunks win, existing-only unknown entry keys survive, incoming unknown-key collisions win, neither input changes, and a later same-mtime incoming legacy entry cannot erase the hash. |
| `test_schema_backward_compatibility_and_invalid_in_memory_data` | A legacy entry missing hash validates; valid unknown entry/chunk keys round-trip through write/load; `merge_vault_entries` and `atomic_write_json` raise `ValueError` for a non-list root, non-object entries/chunks, missing/invalid required fields, invalid hash, non-string unknown keys, non-JSON-compatible values, or non-finite values at any depth. Invalid data aimed at a missing parent creates neither parent nor temp. |
| `test_corrupt_input_backup_then_valid_rebuild_is_unique_and_byte_exact` | Malformed UTF-8/JSON and invalid decoded schema move with `os.rename` to unique regex-matching sibling backups without overwriting a collision; each backup preserves original bytes exactly. After `load_vault` returns `[]`, merge one valid entry and atomically write it; the rebuilt target parses and validates as that one entry. |
| `test_backup_move_failure_leaves_corrupt_target_unchanged` | Monkeypatched `os.rename` raises a sentinel exception; `load_vault` re-raises that same object, leaves the corrupt target byte-for-byte unchanged, creates no backup, and does not rebuild. |
| `test_partial_dump_failure_preserves_target_cleans_temp_and_reraises` | Monkeypatched `json.dump` writes partial text then raises a sentinel; the same exception object propagates, old target bytes remain, and `.<name>.*.tmp` is empty. |
| `test_fsync_failure_preserves_target_cleans_temp_and_reraises` | Monkeypatched `os.fsync` raises a sentinel after dump/flush; the same exception object propagates, old target bytes remain, and the temp glob is empty. |
| `test_chmod_failure_preserves_target_cleans_temp_and_reraises` | With an existing target, monkeypatched `os.chmod` raises a sentinel after close and before replace; the same exception object propagates, old target bytes remain, and the temp glob is empty. |
| `test_replace_failure_preserves_target_cleans_temp_and_reraises` | Monkeypatched `os.replace` raises a sentinel only for temp-to-target replacement; the same exception object propagates, old target bytes remain, and the temp glob is empty. |
| `test_cleanup_failure_preserves_primary_and_records_residual_temp` | Injected replace failure is primary and injected temp `Path.unlink` failure is cleanup: unlink is attempted once; the original replace exception object propagates; its `__notes__` names the residual temp path and cleanup failure; old target bytes remain; exactly that residual temp may remain and is recorded rather than claimed deleted. |
| `test_atomic_temp_order_deterministic_bytes_parent_and_mode` | A valid nested target first creates its missing parents, and the named temp's parent equals the target parent. Instrumented seams prove exact `dump -> flush -> fsync -> close -> chmod -> replace` order for an existing target and `dump -> flush -> fsync -> close -> replace` when absent. Exact UTF-8 bytes show sorted keys, unescaped Unicode, LF-only lines, two-space indent, and one trailing LF; existing permission bits are retained where `os.chmod` is meaningful, while a new target keeps the tempfile mode. |
| `test_builder_rerun_and_same_mtime_content_change` | Tiny temp HTML source and explicit temp `vault_path`; serial pool/tokenizer seams are monkeypatched, run twice unchanged then change cleaned text while restoring mtime; JSON remains one entry and hash/chunks replace once. |
| `test_builder_import_is_side_effect_free` | Isolated subprocess import in empty temp cwd with download/Tk/pool/file-write sentinels; exit 0, no sentinel call, and no vault artifact. |
| `test_owned_writer_has_no_append_mode` | AST inspection of both owned production files finds no `open`/`Path.open`/`NamedTemporaryFile` mode containing `a` and exactly one `os.replace` call in `atomic_write_json`. |

The hash test additionally uses a case alias only after monkeypatching `vault_store.os.path.normcase` to lowercase; tests never assume case folding on POSIX. Native `.`/`..` normalization exercises the real host path functions. The builder test must not use the real network, Tk, multiprocessing, or user folders. Failure tests identify the target replacement by exact source/destination paths so backup `os.rename` and unrelated operations are not intercepted.

## Pristine, RED, GREEN, and comparison evidence

Before edits, from a clean implementation worktree at the frozen SHA, record literal command, SHA, exit code, full summary, and failing test IDs:

```powershell
git rev-parse HEAD
git status --short
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest -p no:cacheprovider tests/test_vault_store.py -q
python -m pytest -p no:cacheprovider tests -q
python -m py_compile Vault_json_creation_from_HTMLs.py vault_store.py
git status --short
```

Observed frozen-tree expectation: the focused command is `4 passed` and exit 0; compile exits 0. In the current environment the full suite exits 2 during collection only for missing `PyQt5` in `tests/test_gui_settings.py` and `tests/test_rag_gui_comprehensive.py`. Record live output rather than copying that environmental baseline if the implementation environment differs. The status must contain no tracked change and no generated vault data.

After replacing only the test file, run:

```powershell
python -m pytest -p no:cacheprovider tests/test_vault_store.py -q
```

RED is mandatory: exactly 17 tests are collected; the four tests `test_load_missing_and_zero_byte_return_empty`, `test_unchanged_rerun_keeps_one_entry_and_object_value`, `test_changed_mtime_with_unchanged_or_absent_hash_replaces_once`, and `test_owned_writer_has_no_append_mode` are expected to pass against the frozen implementation, while the other 13 named tests are expected to fail. Record the actual exit code, pass/fail count, and every failing ID; if the live result differs, stop and explain the frozen-code/test mismatch before production edits. Commit tests only after recording RED:

```text
test(#15): specify atomic vault behavior
```

Implement only the locked production changes, then run:

```powershell
python -m pytest -p no:cacheprovider tests/test_vault_store.py -q
python -m py_compile Vault_json_creation_from_HTMLs.py vault_store.py
git grep -n -E 'mode\s*=\s*["'']a|open\([^)]*,\s*["'']a' -- Vault_json_creation_from_HTMLs.py vault_store.py
git diff --check
```

GREEN requires exactly `17 passed`; compile and diff checks exit 0; the append `git grep` exits 1 with no output, which is the expected no-match result. Any grep output or any other exit code is failure. Commit production only after GREEN:

```text
fix(#15): atomic-write vault json
```

Then run the unchanged comparison block and record exact exit codes, full summaries, and failing test IDs:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest -p no:cacheprovider tests/test_vault_store.py -q
python -m pytest -p no:cacheprovider tests -q
python -m py_compile Vault_json_creation_from_HTMLs.py vault_store.py
gh issue list --repo sriharshaguthikonda/easy-local-rag --state open
git status --short
```

The focused suite has exactly 17 passes. The full suite has no new failure ID or classification compared with pristine; a pre-existing missing-PyQt5 collection error may remain. Generated `vault.json`, invalid backups, temp files, `__pycache__`, and pytest cache must not be staged or included in closure evidence.

## Threat and rollback boundaries

The trust boundary is local disk content into `load_vault`; schema validation prevents malformed persisted data from silently replacing a valid array. Same-directory temp plus fsync plus replace mitigates partial writes. Backup names contain no source content, and warnings expose paths only to the local console. There is no new package/supply-chain surface and no secret or network input added. The runtime Punkt fallback is pre-existing behavior moved out of import time.

Rollback is reverse-order `git revert` of accepted review-fix commits, then the production commit, then the test commit. Do not restore append mode. Before rollback, preserve any user vault outside Git; the repository rollback never deletes generated data or corruption backups. If the production commit is reverted while the builder remains maintained, restore the last known-good pre-change vault manually from the operator's backup and stop using the builder until the atomic fix is restored.

## Review, activation, commits, and closure

The sequential lifecycle is exactly: **ChatGPT review -> GSD checker -> corrector -> docs merge -> separate activation PR merge -> implementer -> reviewer -> fixer -> verifier -> orchestrator merge/close**.

1. ChatGPT reviews this exact docs head and posts SHA-bound PASS or numbered blockers.
2. GSD checker independently reviews the same head and posts SHA-bound PASS or numbered blockers.
3. The corrector changes only the three packet-owned docs, records every disposition and correction SHA, and repeats both reviews until both PASS the same final head.
4. The orchestrator merges the docs PR and records its immutable merge SHA. Planning remains non-active after that merge and throughout the separate activation PR review.
5. After the docs merge, the orchestrator opens and merges a separate activation PR naming the reviewed packet SHA, docs merge SHA, frozen GUI SHA, implementation branch, and worker ownership. Only that merged PR constitutes activation; its URL and immutable merge SHA are recorded before code work begins.
6. The implementer records pristine and RED evidence, commits exactly `test(#15): specify atomic vault behavior`, implements, records GREEN/comparison evidence, and commits exactly `fix(#15): atomic-write vault json`. No squash between these two TDD commits before review.
7. The code reviewer reviews the initial production head. The fixer makes one atomic `fix(#15): address accepted review finding` commit per accepted finding; rejected findings get a written reason.
8. The verifier runs every packet command on the final code SHA and reports PASS/BLOCKED without editing code.
9. The orchestrator alone merges the code PR, posts reciprocal evidence links to the issue/canonical packet, updates the roadmap, and closes #15 only when every closure item below exists.

Approval stops are blocking: no docs merge without exact-head ChatGPT and GSD PASS; no code work until the separate activation PR has merged and its URL and immutable merge SHA are recorded; no code merge/issue closure without final reviewer dispositions and verifier PASS. Planning docs are never implementation evidence.

Closure evidence names the GitHub issue, canonical plan, packet, final reviewed docs SHA and docs merge SHA, the separate merged activation PR URL and its immutable merge SHA, frozen GUI SHA, exact test commit, exact initial production commit, every accepted-finding fix SHA or `none accepted`, final verifier SHA, code PR URL and immutable merge SHA, pristine/RED/GREEN/full-suite command outputs, append `git grep` exit 1/no output, `git diff --check`, and clean generated-data status. It explicitly names and proves both `test_changed_mtime_with_unchanged_or_absent_hash_replaces_once` and `test_same_mtime_changed_hash_replaces_once`, plus unchanged rerun/object stability, non-mutating merge winners, schema/backward compatibility, byte-exact corruption backup followed by valid rebuild, backup-move failure preservation, same-directory temp and exact event order, dump/fsync/chmod/replace primary-failure preservation and cleanup, cleanup-failure residual recording, builder integration, side-effect-free import, deterministic serialization/mode behavior, and no append mode. It states that #8 remains open and receives no closure credit from #15.
