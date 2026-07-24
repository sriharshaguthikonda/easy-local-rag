# Issue #15 Atomic Vault Write Implementation Packet

**Goal:** make the maintained HTML vault builder idempotent and make every `vault.json` replacement atomic without losing the previous valid target on write failure.

[GitHub Issue #15](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15) · [canonical plan](../../issues/ISSUE-015-atomic-vault-write.md) · [roadmap ledger](../../issues/README.md)

**Docs packet lineage:** `codex/issue-015-plan`, based on `origin/main` at `07fdddad6d2377ed3d147f7ba0de8a65c83ebdff`. Git history carries the packet commits; the immutable reviewed docs-merge SHA is required before activation branch `codex/issue-015-activate` may open its PR into `main`.

**Code lineage:** create branch `codex/fix-issue-15` exactly at frozen target head `619365224f6db3770d3369fd84a295589699e513`; its code PR base is exactly `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`. That SHA is the merge of [PR #33](https://github.com/sriharshaguthikonda/easy-local-rag/pull/33) and is also the exact frozen `origin/GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` head. Immediately before implementation work and again before code merge, fetch that remote branch and stop if its head is not exactly the frozen SHA. Never merge or rebase main/docs into the code lineage.

**Mode:** maintained path. `Vault_json_creation_from_HTMLs.py` remains runnable. Issue #8 is still open: #15 may retain the already-landed `EASY_RAG_VAULT_SOURCE_DIR` override with the existing folder-picker fallback, but it must not claim #8 closure or edit #8-owned configuration/docs. No dependency change is allowed. The only schema evolution is the canonical gate's additive `content_hash`; legacy entries without it remain readable and are upgraded when their source is next processed.

**Worker-owned code/test files:** `vault_store.py`, `Vault_json_creation_from_HTMLs.py`, `tests/test_vault_store.py`. `.gitignore` is verification-only because `*.json` and `vault.json` are already ignored. Do not modify another source, test, configuration, dependency, generated vault, or documentation file.

**Runtime floor:** Python 3.11 or newer is mandatory. The exact preflight below runs before pristine/RED work and again before final verification; any failure stops the relevant phase. This floor makes `BaseException.add_note` part of the locked failure contract without a compatibility fallback.

## Authority and source coverage

GitHub remains authoritative for live state and owner decisions; the canonical plan fixes scope and closure gates; the roadmap fixes sequencing/status; this packet locks implementation detail without weakening either source.

| Source | Required item | Packet coverage |
|---|---|---|
| GitHub issue and owner plan comment | Read existing list, normalized `file_name` identity, unchanged keep, changed replace, corruption backup, same-directory temp/replace, no append | Locked below in schema, merge, failure, and tests |
| [Owner two-commit decision](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15#issuecomment-5070864427) | Exactly two named TDD commits, separate per-finding review-fix commits, no squash before review, and no activation before a separate activation PR merges | Commit names and lifecycle are locked below without changing any other gate |
| [Original ten-finding disposition](https://github.com/sriharshaguthikonda/easy-local-rag/pull/34#issuecomment-5071265859) | Immutable mapping from the first ChatGPT/GSD union to correction head `21249fa5f34c43559d21c2740e50e0a5eb78187f` | Retained as review-history authority; later exact-head reviews remain additive |
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
- at the same mtime, an existing hash plus an incoming missing hash returns that exact existing hashed entry object unchanged at its original position. It ignores all incoming `file_name`, `modification_time`, `chunks`, and unknown-key changes, retains the existing hash/chunks/unknown keys exactly, and leaves both input lists and every nested object unchanged.

Replacement and hash upgrade construct a new entry without mutating either input: incoming `file_name`, `modification_time`, `chunks`, and `content_hash` (when present) are canonical; existing-only unknown entry keys are copied forward; incoming unknown-key collisions overwrite existing unknown values. On a changed-mtime replacement, omission of incoming `content_hash` does not copy the stale existing canonical hash. A same-version keep—including the same-mtime existing-hash/incoming-missing-hash no-downgrade case—is byte/object-value stable and returns the existing entry object at the same position. Tests deep-copy both inputs and assert equality after every merge, assert identity and exact existing chunks/unknown keys in the no-downgrade keep, and assert canonical/chunk/unknown-key winners only on replacement and hash upgrade.

The builder processes each discovered HTML-family file, computes `content_hash`, and lets the merge helper decide unchanged versus replacement. One unchanged rerun remains one entry; changed cleaned content with an unchanged filesystem mtime replaces exactly once.

## Locked corruption, serialization, and failure contract

`load_vault` returns `[]` only for a missing or zero-byte target. Valid schema returns newly decoded data. It calls `_validate_vault_data` after decoding. `UnicodeDecodeError`, `json.JSONDecodeError`, or schema `ValueError` means corrupt on-disk input; other read/stat permission or I/O failures propagate and are not mislabeled as corruption.

Corrupt input is moved before recovery to the first unused sibling named `vault.json.invalid.<UTC-YYYYMMDDTHHMMSSffffffZ>.<pid>[.<counter>].bak`; UTC text is exactly `datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")` and pid is `os.getpid()`. The unsuffixed candidate is first and collision counters start at `.1`. `_invalid_backup_path` checks candidates deterministically and the move primitive is exactly `os.rename(vault_path, backup_path)`, never `os.replace`, so the implementation does not intentionally overwrite an existing backup. This packet does not add a concurrent-reader/writer reservation protocol. The backup preserves corrupt bytes exactly, `load_vault` prints one warning naming it, and returns `[]`. Backup-move failure propagates the original exception and leaves the corrupt target byte-for-byte unchanged.

`atomic_write_json` first calls `_validate_vault_data(data)` without mutating `data`. Only after validation succeeds does it always call `vault_path.parent.mkdir(parents=True, exist_ok=True)`, before temp creation; invalid data therefore cannot create a missing parent. It snapshots an existing target's permission bits with `stat.S_IMODE(vault_path.stat().st_mode)`. It then creates one sibling temp through `tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=f".{vault_path.name}.", suffix=".tmp", dir=vault_path.parent, delete=False)`. `handle.name` is the cleanup path; the temp is in exactly `vault_path.parent`, so Windows `os.replace` remains same-volume.

Serialization is deterministic: one `json.dump(data, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)`, followed by one explicit `handle.write("\n")`. The observable success order is exactly `dump -> flush -> os.fsync(handle.fileno()) -> close -> os.chmod(temp_path, saved_mode)` only when the target existed -> `os.replace(temp_path, vault_path)`. A missing target retains the platform's secure tempfile mode and skips chmod. `os.replace` is called exactly once and only after the handle is closed, which is required for Windows. Directory fsync is not added: the maintained target is Windows, and the canonical gate requires atomic visibility through same-volume `os.replace`, not a new cross-platform power-loss durability contract.

No append open and no second write branch may exist. On JSON/UTF-8 encode-write/flush/fsync/close/chmod/replace failure, the caught primary exception remains authoritative and the implementation attempts to close any still-open handle and unlink the named temp if it exists; replacement has not completed, so the old target remains byte-for-byte unchanged. If close or unlink cleanup fails, that cleanup exception never replaces the primary exception. Under mandatory Python 3.11+, the implementation uses `BaseException.add_note` to record the residual temp path and cleanup failure on the primary exception, then re-raises that same primary exception object. A residual temp may physically remain when unlink itself fails; the cleanup-failure test requires and records that residual instead of claiming successful deletion.

`Vault_json_creation_from_HTMLs.py` has no import-time download, tokenizer load, Tk window, pool, file scan, or vault write. `_get_sentence_tokenizer()` first loads local Punkt; only a runtime `LookupError` may trigger the existing quiet Punkt download and retry. `main()` alone applies #8's current rule: use an existing `EASY_RAG_VAULT_SOURCE_DIR`, otherwise open the existing picker; cancel means return without writing. `convert_html_to_json()` never opens UI and propagates storage/source/tokenizer errors with a nonzero CLI exit.

## TDD test matrix

Replace `tests/test_vault_store.py` first. It contains exactly these 17 named test functions, with no parametrization that changes the reported count. Tests use only `tmp_path`, `monkeypatch`, stdlib, and installed project dependencies:

| Test ID | Exact proof |
|---|---|
| `test_load_missing_and_zero_byte_return_empty` | Missing and zero-byte paths both return `[]`; neither creates a backup. |
| `test_unchanged_rerun_keeps_one_entry_and_object_value` | Merge then write/read twice with one normalized identity and identical mtime/hash; result length is one, the merge returns the existing entry object at its original position, its object value and deterministic serialized bytes are unchanged, and both input lists still equal deep copies. |
| `test_changed_mtime_with_unchanged_or_absent_hash_replaces_once` | Two independent merges use one stable normalized identity represented with a native `.`/`..` alias and changed `modification_time`: one keeps the same hash and one has hashes absent on both versions. Each remains one entry; incoming canonical fields/chunks win exactly once at the original position. This is independent of hash-change behavior. |
| `test_same_mtime_changed_hash_replaces_once` | One stable normalized identity with unchanged mtime and different 64-hex `content_hash` remains one entry; incoming canonical fields/chunks/hash win exactly once at the original position. This is independent of mtime-change behavior. |
| `test_legacy_hash_upgrade_merge_winners_and_no_mutation` | A same-mtime legacy entry missing hash upgrades from an incoming hash; incoming canonical fields/chunks win, existing-only unknown entry keys survive, incoming unknown-key collisions win, and neither input changes. Then, at the same mtime, an existing hashed entry plus an incoming missing-hash entry with deliberately different canonical fields/chunks/unknown keys returns the exact existing hashed object at its original position; identity, hash, chunks, and unknown keys are exact, incoming changes are ignored, and both inputs still equal deep copies. |
| `test_schema_backward_compatibility_and_invalid_in_memory_data` | A legacy entry missing hash validates; valid unknown entry/chunk keys round-trip through write/load; `merge_vault_entries` and `atomic_write_json` raise `ValueError` for a non-list root, non-object entries/chunks, missing/invalid required fields, invalid hash, non-string unknown keys, non-JSON-compatible values, or non-finite values at any depth. Invalid data aimed at a missing parent creates neither parent nor temp. |
| `test_corrupt_input_backup_then_valid_rebuild_is_unique_and_byte_exact` | Malformed UTF-8/JSON and invalid decoded schema move with `os.rename` to unique regex-matching sibling backups without overwriting a collision; each backup preserves original bytes exactly. After `load_vault` returns `[]`, merge one valid entry and atomically write it; the rebuilt target parses and validates as that one entry. |
| `test_backup_move_failure_leaves_corrupt_target_unchanged` | Monkeypatched `os.rename` raises a sentinel exception; `load_vault` re-raises that same object, leaves the corrupt target byte-for-byte unchanged, creates no backup, and does not rebuild. |
| `test_dump_encode_write_flush_and_close_failures_preserve_target` | One non-parametrized test runs four sequential, independently reset subcases for partial `json.dump`, UTF-8 encode/write, explicit flush, and explicit close failure. Each injects one sentinel at the locked seam/event position, proves the same exception object propagates, old target bytes remain, no chmod/replace follows, both inputs remain unchanged, and the temp glob is empty after successful cleanup. |
| `test_fsync_failure_preserves_target_cleans_temp_and_reraises` | Monkeypatched `os.fsync` raises a sentinel after dump/flush; the same exception object propagates, old target bytes remain, and the temp glob is empty. |
| `test_chmod_failure_preserves_target_cleans_temp_and_reraises` | With an existing target, monkeypatched `os.chmod` raises a sentinel after close and before replace; the same exception object propagates, old target bytes remain, and the temp glob is empty. |
| `test_replace_failure_preserves_target_cleans_temp_and_reraises` | Monkeypatched `os.replace` raises a sentinel only for temp-to-target replacement; the same exception object propagates, old target bytes remain, and the temp glob is empty. |
| `test_cleanup_failure_preserves_primary_and_records_residual_temp` | Injected replace failure is primary and injected temp `Path.unlink` failure is cleanup: unlink is attempted once; the original replace exception object propagates; its `__notes__` names the residual temp path and cleanup failure; old target bytes remain; exactly that residual temp may remain and is recorded rather than claimed deleted. |
| `test_atomic_temp_order_deterministic_bytes_parent_and_mode` | A valid nested target first creates its missing parents, and the named temp's parent equals the target parent. Instrumented seams prove exact `dump -> flush -> fsync -> close -> chmod -> replace` order for an existing target and `dump -> flush -> fsync -> close -> replace` when absent. Exact UTF-8 bytes show sorted keys, unescaped Unicode, LF-only lines, two-space indent, and one trailing LF; existing permission bits are retained where `os.chmod` is meaningful, while a new target keeps the tempfile mode. |
| `test_builder_rerun_and_same_mtime_content_change` | Tiny temp HTML source and explicit temp `vault_path`; serial pool/tokenizer seams are monkeypatched, run twice unchanged then change cleaned text while restoring mtime; JSON remains one entry and hash/chunks replace once. |
| `test_builder_import_is_side_effect_free` | Isolated subprocess import in empty temp cwd with download/Tk/pool/file-write sentinels; exit 0, no sentinel call, and no vault artifact. |
| `test_owned_writer_has_no_append_mode` | AST inspection of both owned production files finds no `open`/`Path.open`/`NamedTemporaryFile` mode containing `a` and exactly one `os.replace` call in `atomic_write_json`. |

The hash test additionally uses a case alias only after monkeypatching `vault_store.os.path.normcase` to lowercase; tests never assume case folding on POSIX. Native `.`/`..` normalization exercises the real host path functions. The builder test must not use the real network, Tk, multiprocessing, or user folders. Failure tests identify the target replacement by exact source/destination paths so backup `os.rename` and unrelated operations are not intercepted.

The combined dump/encode-write/flush/close test uses a fresh target, payload, sentinel object, event list, and temp-handle wrapper for each sequential subcase; it is one ordinary test function, not parametrized. A `json.dump` wrapper injects partial dump failure after writing one fragment. The handle wrapper injects UTF-8 encode/write failure from its first serialization `write`. Its explicit `flush` method injects flush failure. For close failure only, its explicit `close` first closes the delegated real handle and then raises the close sentinel; this makes the close event observable while leaving the native Windows handle physically closed so cleanup can unlink the temp. Instrumented `json.dump`, `flush`, `os.fsync`, `close`, `os.chmod`, `os.replace`, and temp unlink seams assert these exact primary-event prefixes: dump `dump -> dump-fail`; encode/write `dump -> encode-write-fail`; flush `dump -> flush-fail`; close `dump -> flush -> fsync -> close-fail`. Cleanup close/unlink events may follow a primary failure as physically applicable, but no later success-protocol event may precede it and neither chmod nor replace may occur. Each subcase asserts the raised object `is` its sentinel, the old target is byte-exact, the temp glob is empty after cleanup, and the input equals its deep copy. The separate cleanup-failure test remains the only case allowed to assert a residual temp and its `__notes__`.

## Pristine, RED, GREEN, and comparison evidence

After the activation merge and before any implementation edit, create `codex/fix-issue-15` directly at the frozen SHA. From its clean worktree, run the frozen-base and Python preflight below first. Record every literal command, SHA/version, exit code, full summary, and failing test ID. A fetch, remote-head, branch/head, or Python check failure stops before edits:

```powershell
$targetBranch = 'GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs'
$frozenSha = '619365224f6db3770d3369fd84a295589699e513'
git fetch origin $targetBranch
if ($LASTEXITCODE -ne 0) { throw 'Cannot fetch frozen GUI base; stop before edits.' }
$remoteBase = (git rev-parse "refs/remotes/origin/$targetBranch").Trim()
if ($LASTEXITCODE -ne 0 -or $remoteBase -ne $frozenSha) { throw "Remote GUI base is $remoteBase, expected $frozenSha; stop before edits." }
if ((git branch --show-current).Trim() -ne 'codex/fix-issue-15') { throw 'Wrong implementation branch; stop before edits.' }
if ((git rev-parse HEAD).Trim() -ne $frozenSha) { throw 'Implementation branch was not created at the frozen SHA; stop before edits.' }
python -c "import sys; print(sys.version); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required; stop before edits.' }
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
python -c "import sys; print(sys.version); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required; stop before RED.' }
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

Then run the code-only final comparison block and record exact exit codes, full summaries, and failing test IDs. The Python preflight is mandatory again and stops final verification on failure:

```powershell
python -c "import sys; print(sys.version); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required; stop final verification.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest -p no:cacheprovider tests/test_vault_store.py -q
python -m pytest -p no:cacheprovider tests -q
python -m py_compile Vault_json_creation_from_HTMLs.py vault_store.py
git status --short
```

The focused suite has exactly 17 passes. The full suite has no new failure ID or classification compared with pristine; a pre-existing missing-PyQt5 collection error may remain. Live GitHub issue state is not part of this code-correctness comparison. Generated `vault.json`, invalid backups, temp files, `__pycache__`, and pytest cache must not be staged or included in closure evidence.

Live issue state is an orchestrator-owned closure observation, separate from code correctness. Run this preflight and record command availability, auth exit, issue-list exit, and output. Missing `gh`, auth failure, or network failure is recorded but does not change a passing code verdict:

```powershell
$ghCommand = Get-Command gh -ErrorAction SilentlyContinue
if ($null -eq $ghCommand) {
    Write-Warning 'gh is unavailable; record the observation and continue code verification.'
} else {
    gh auth status
    $ghAuthExit = $LASTEXITCODE
    if ($ghAuthExit -ne 0) {
        Write-Warning "gh auth status failed with exit $ghAuthExit; record it and continue code verification."
    } else {
        gh issue list --repo sriharshaguthikonda/easy-local-rag --state open
        $ghIssueExit = $LASTEXITCODE
        if ($ghIssueExit -ne 0) {
            Write-Warning "Live issue observation failed with exit $ghIssueExit; record it and continue code verification."
        }
    }
}
```

Before any GitHub merge, comment, roadmap update, or issue close, working authenticated GitHub access is mandatory. Immediately before the code merge, also refetch and recheck the frozen base and current PR base; any failure stops orchestration:

```powershell
$ghCommand = Get-Command gh -ErrorAction SilentlyContinue
if ($null -eq $ghCommand) { throw 'gh is required before merge/comment/close.' }
gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Authenticated GitHub access is required before merge/comment/close.' }
$targetBranch = 'GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs'
$frozenSha = '619365224f6db3770d3369fd84a295589699e513'
git fetch origin $targetBranch
if ($LASTEXITCODE -ne 0) { throw 'Cannot fetch frozen GUI base; stop before merge.' }
$remoteBase = (git rev-parse "refs/remotes/origin/$targetBranch").Trim()
if ($LASTEXITCODE -ne 0 -or $remoteBase -ne $frozenSha) { throw "Remote GUI base is $remoteBase, expected $frozenSha; stop before merge." }
$codePrBase = (gh pr view --json baseRefName --jq '.baseRefName').Trim()
if ($LASTEXITCODE -ne 0 -or $codePrBase -ne $targetBranch) { throw "Code PR base is $codePrBase, expected $targetBranch; stop before merge." }
```

After this block passes, the only permitted code-PR merge option is GitHub **Create a merge commit**.

## Threat and rollback boundaries

The trust boundary is local disk content into `load_vault`; schema validation prevents malformed persisted data from silently replacing a valid array. Same-directory temp plus fsync plus replace mitigates partial writes. Backup names contain no source content, and warnings expose paths only to the local console. There is no new package/supply-chain surface and no secret or network input added. The runtime Punkt fallback is pre-existing behavior moved out of import time.

The default full rollback after merge is `git revert -m 1` of the immutable code-PR merge commit on the GUI branch. Before merge, or when an explicitly reviewed surgical rollback is required, revert the preserved child commits in reverse order: accepted review-fix commits, production commit, then test commit. GitHub **Create a merge commit** keeps every exact child SHA reachable for that evidence and rollback; squash/rebase would destroy this contract and is forbidden. Do not restore append mode. Before rollback, preserve any user vault outside Git; the repository rollback never deletes generated data or corruption backups. If the production behavior is removed while the builder remains maintained, restore the last known-good pre-change vault manually from the operator's backup and stop using the builder until the atomic fix is restored.

## Review, activation, commits, and closure

The sequential lifecycle is exactly: **ChatGPT review -> GSD checker -> corrector -> docs merge -> separate activation PR merge -> implementer -> reviewer -> fixer -> verifier -> orchestrator merge/close**.

Review history is immutable: the [first ChatGPT review](https://github.com/sriharshaguthikonda/easy-local-rag/pull/34#issuecomment-5070780534), [first GSD check](https://github.com/sriharshaguthikonda/easy-local-rag/pull/34#issuecomment-5070864055), and [original ten-finding disposition](https://github.com/sriharshaguthikonda/easy-local-rag/pull/34#issuecomment-5071265859) lead to the [second ChatGPT review](https://github.com/sriharshaguthikonda/easy-local-rag/pull/34#issuecomment-5071169689) and [second GSD check](https://github.com/sriharshaguthikonda/easy-local-rag/pull/34#issuecomment-5071266378). The last two are blockers on `21249fa5f34c43559d21c2740e50e0a5eb78187f`, not PASS evidence.

1. ChatGPT reviews the final exact docs head and posts PASS naming that full SHA; retain the immutable PASS comment URL and numeric comment ID.
2. GSD checker independently reviews the same final head and posts PASS naming that full SHA; retain its immutable PASS comment URL and numeric comment ID.
3. The corrector changes only the three packet-owned docs, records every disposition and correction SHA, and repeats both reviews until both PASS the same final head. Immediately after that final correction commit, the orchestrator—and no other owner—updates PR #34's body to name the final exact head and exactly 17 test functions. Updating that PR body is mandatory; there is no substitute in docs, comments, or later closure evidence.
4. The orchestrator merges the docs PR and records its immutable merge SHA. Planning remains non-active after that merge and throughout the separate activation PR review.
5. After the docs merge, branch `codex/issue-015-activate` opens a PR into `main`. Its ownership is only `docs/issues/ISSUE-015-atomic-vault-write.md` and `docs/issues/README.md`; no packet, code, test, configuration, or other file may change. The activation commit changes canonical `PLANNING` to `ACTIVE`, roadmap #15 `planning` to `active`, and roadmap `Active issue` `none` to `#15`. It records the actual final reviewed packet SHA, immutable docs merge SHA, frozen GUI SHA `619365224f6db3770d3369fd84a295589699e513`, implementation branch `codex/fix-issue-15`, and worker ownership of `vault_store.py`, `Vault_json_creation_from_HTMLs.py`, and `tests/test_vault_store.py`. Only that PR's GitHub merge commit SHA activates work; its URL and merge SHA are recorded before branch creation or code work.
6. After activation, create `codex/fix-issue-15` exactly at `619365224f6db3770d3369fd84a295589699e513`, run the frozen-base/Python preflight, record pristine and RED evidence, and commit exactly `test(#15): specify atomic vault behavior`. Implement only the locked production scope, record GREEN/comparison evidence, and commit exactly `fix(#15): atomic-write vault json`.
7. The code PR from `codex/fix-issue-15` targets exactly `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`. The code reviewer reviews the initial production head. The fixer makes one atomic `fix(#15): address accepted review finding` commit per accepted finding; rejected findings get a written reason. An immutable disposition comment URL and numeric comment ID must name the exact final code SHA and map every finding to its fix SHA or rejection reason.
8. The verifier runs every packet command on that exact final code SHA without editing code. An immutable verifier PASS comment URL and numeric comment ID must name the same final code SHA.
9. Immediately before code merge, the orchestrator repeats the authenticated-GitHub and frozen-remote-base stop guards below and confirms the PR base exactly. The PR is merged using GitHub **Create a merge commit** only—never squash or rebase—so the exact test, production, and accepted review-fix SHAs remain child commits. The orchestrator then posts reciprocal evidence links, updates the roadmap, and closes #15 only when every closure item below exists.

Approval stops are blocking: no docs merge without the two final exact-head PASS URLs/IDs; no code work until the separate activation PR has merged and its URL and immutable merge SHA are recorded; no code merge/issue closure without the SHA-bound code-review disposition URL/ID and verifier PASS URL/ID. Planning docs are never implementation evidence.

Closure evidence names the GitHub issue, canonical plan, packet, final reviewed docs SHA, both final exact-head plan PASS URLs/IDs, docs merge SHA, activation PR URL, activation commit SHA, activation merge SHA, frozen GUI SHA, implementation branch, exact test commit, exact initial production commit, every accepted-finding fix SHA or `none accepted`, exact final code SHA, immutable code-review disposition URL/ID and verifier PASS URL/ID that each name that final code SHA, code PR URL and immutable merge SHA, pristine/RED/GREEN/full-suite command outputs, Python 3.11+ preflights, append `git grep` exit 1/no output, `git diff --check`, and clean generated-data status. It explicitly names and proves both `test_changed_mtime_with_unchanged_or_absent_hash_replaces_once` and `test_same_mtime_changed_hash_replaces_once`, plus unchanged rerun/object stability, hashed-entry no-downgrade identity/exact chunks/unknown keys/no-input-mutation, merge winners, schema/backward compatibility, byte-exact corruption backup followed by valid rebuild, backup-move failure preservation, same-directory temp and exact event order, dump/UTF-8 encode-write/flush/fsync/close/chmod/replace primary-failure preservation and cleanup, cleanup-failure residual recording, builder integration, side-effect-free import, deterministic serialization/mode behavior, and no append mode. It records all final commit and merge SHAs and states that #8 remains open and receives no closure credit from #15.
