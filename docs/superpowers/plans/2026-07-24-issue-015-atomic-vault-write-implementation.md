# Issue #15 Atomic Vault Write Implementation Packet

**Goal:** make the maintained HTML vault builder idempotent and make every `vault.json` replacement atomic without losing the previous valid target on write failure.

[GitHub Issue #15](https://github.com/sriharshaguthikonda/easy-local-rag/issues/15) · [canonical plan](../../issues/ISSUE-015-atomic-vault-write.md) · [roadmap ledger](../../issues/README.md)

**Docs packet lineage:** `codex/issue-015-plan`, based on `origin/main` at `07fdddad6d2377ed3d147f7ba0de8a65c83ebdff`. Git history carries the packet commits; the immutable reviewed docs-merge SHA is required before activation.

**Code lineage:** create the implementation branch directly at frozen target head `619365224f6db3770d3369fd84a295589699e513`, target branch `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`. That SHA is the merge of [PR #33](https://github.com/sriharshaguthikonda/easy-local-rag/pull/33) and is also the exact `origin/GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` head used for this packet. Never merge or rebase main/docs into the code lineage.

**Mode:** maintained path. `Vault_json_creation_from_HTMLs.py` remains runnable. Issue #8 is still open: #15 may retain the already-landed `EASY_RAG_VAULT_SOURCE_DIR` override with the existing folder-picker fallback, but it must not claim #8 closure or edit #8-owned configuration/docs. No dependency change is allowed. The only schema evolution is the canonical gate's additive `content_hash`; legacy entries without it remain readable and are upgraded when their source is next processed.

**Worker-owned code/test files:** `vault_store.py`, `Vault_json_creation_from_HTMLs.py`, `tests/test_vault_store.py`. `.gitignore` is verification-only because `*.json` and `vault.json` are already ignored. Do not modify another source, test, configuration, dependency, generated vault, or documentation file.

## Authority and source coverage

GitHub remains authoritative for live state and owner decisions; the canonical plan fixes scope and closure gates; the roadmap fixes sequencing/status; this packet locks implementation detail without weakening either source.

| Source | Required item | Packet coverage |
|---|---|---|
| GitHub issue and owner plan comment | Read existing list, normalized `file_name` identity, unchanged keep, changed replace, corruption backup, same-directory temp/replace, no append | Locked below in schema, merge, failure, and tests |
| Canonical #15 plan | Maintained/retirement choice, stable identity/version semantics, atomic cleanup, four named regressions, rollback | Maintained mode and every named regression locked below |
| Roadmap ledger | #9 closes before #15; #8 fallback does not close #8; one active issue maximum | #9 evidence frozen; #15 remains planning with no active slot until activation |
| Frozen GUI code/tests | Existing helper signatures, builder wiring, current four-test baseline, existing #8 env/picker fallback | Caller map, ownership, TDD baseline, and exact changes locked below |

There is no phase `RESEARCH.md`, `CONTEXT.md`, or requirement-ID artifact for this legacy issue packet. Live code inspection at the frozen SHA supplies implementation evidence; no source item is omitted or deferred by this packet.

## Frozen caller and entry-point map

Derived from the frozen SHA with:

```powershell
git grep -n -E "load_vault|merge_vault_entries|atomic_write_json|convert_html_to_json|Vault_json_creation_from_HTMLs|vault\.json" 619365224f6db3770d3369fd84a295589699e513 -- '*.py' '*.md'
```

- `vault_store.py:9`, `:33`, and `:58` define the three public storage helpers. Their only production importer is `Vault_json_creation_from_HTMLs.py:12`; their only other importer is `tests/test_vault_store.py:4`.
- `Vault_json_creation_from_HTMLs.py:91` defines `convert_html_to_json`; it loads at `:93`, merges at `:133`, atomically writes at `:134`, and invokes `main()` only under the `__main__` guard at `:154-155`.
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

`vault_store.py` owns exactly three private behaviors: `_validate_vault_data(data)`, `_normalized_file_identity(file_name)`, and `_invalid_backup_path(vault_path)`. Private names may not become public exports. `Vault_json_creation_from_HTMLs.py` owns `_get_sentence_tokenizer()` for lazy tokenizer loading; existing `clean_text`, `extract_text_from_html`, `generate_chunk_id`, `split_into_chunks`, `convert_html_to_json`, and `main` remain public by compatibility.

`path`/`vault_path` accepts strings and `os.PathLike`. Relative output paths resolve from the current working directory. The builder expands `~`, resolves the source directory to an absolute path without requiring #8 changes, raises `FileNotFoundError` for a missing source and `NotADirectoryError` for a non-directory, and persists each `file_name` as `os.path.normpath(str(Path(file_path).resolve()))`. Storage identity is `os.path.normcase(os.path.abspath(os.path.normpath(file_name)))`; therefore Windows case/separator aliases identify one file, while moving a file creates a new identity. Empty/non-string identities are invalid data.

## Locked vault schema, merge order, and limits

The on-disk document is exactly one JSON array. Every entry is an object requiring:

- non-empty string `file_name`, persisted by the builder as the normalized absolute path;
- finite numeric `modification_time`, excluding `bool`;
- list `chunks`; each generated chunk is an object with lowercase 64-hex SHA-256 string `id` and string `text`;
- optional lowercase 64-hex SHA-256 string `content_hash`; the builder always emits it as SHA-256 of the UTF-8 cleaned full document text.

Unknown entry/chunk keys with string names and JSON-serializable values are preserved for compatibility but are not interpreted. A missing `content_hash` is valid legacy data. There are no new issue-specific byte, entry, or chunk limits: this local utility continues to load and serialize the array in memory, bounded by Python memory and available same-volume disk. Non-finite floats are invalid. A root other than a list, a non-object entry, a missing/invalid required field, an invalid optional hash, malformed UTF-8, or malformed JSON is corrupt input.

Merge identity never contains mtime, chunk IDs, or hashes. Existing duplicate identities collapse deterministically: the last existing value wins while retaining the identity's first position. Incoming duplicates behave the same. New identities append in incoming first-seen order. For an existing identity:

- same mtime and same hash keeps the existing entry unchanged;
- same mtime with both hashes absent keeps the legacy entry;
- an incoming hash upgrades an existing missing hash;
- a changed mtime or changed non-null hash replaces the entry in its existing position;
- an incoming missing hash does not downgrade an existing hash when mtime is unchanged.

The builder processes each discovered HTML-family file, computes `content_hash`, and lets the merge helper decide unchanged versus replacement. One unchanged rerun remains one entry; changed cleaned content with an unchanged filesystem mtime replaces exactly once.

## Locked corruption, serialization, and failure contract

`load_vault` returns `[]` only for a missing or zero-byte target. Valid schema returns newly decoded data. Corrupt input is moved before recovery to a sibling named `vault.json.invalid.<UTC-YYYYMMDDTHHMMSSffffffZ>.<pid>[.<counter>].bak`; collision counters start at `.1`, never overwrite a backup, and preserve the corrupt bytes exactly. It prints one warning naming the backup and returns `[]`. Backup-move failure propagates and leaves the target untouched. Other read/stat permission or I/O failures propagate; they are not mislabeled as corruption.

`atomic_write_json` validates before creating a temp file. It creates one sibling temp named `.<target-name>.<random>.tmp` through `tempfile.NamedTemporaryFile(delete=False)`, UTF-8 text mode with `newline="\n"`. Serialization is deterministic: `ensure_ascii=False`, `allow_nan=False`, `sort_keys=True`, `indent=2`, and exactly one trailing LF. The function flushes and calls `os.fsync` on the open temp file, closes it, copies the existing target's permission mode bits when a target exists (otherwise retains the platform's secure tempfile mode), then calls `os.replace(temp, target)` exactly once. The temp is on the same directory/volume as the target.

No append open and no second write branch may exist. Directory fsync is not added: the maintained target is Windows, and the canonical gate requires atomic visibility through same-volume `os.replace`, not a new cross-platform power-loss durability contract. On JSON/encode/flush/fsync/chmod/replace failure, propagate the original exception, delete the named temp if it still exists, and leave the old target byte-for-byte unchanged whenever replacement did not complete. Cleanup failure must not mask the original exception. Parent-directory creation may occur before temp creation; no target file is created until replace.

`Vault_json_creation_from_HTMLs.py` has no import-time download, tokenizer load, Tk window, pool, file scan, or vault write. `_get_sentence_tokenizer()` first loads local Punkt; only a runtime `LookupError` may trigger the existing quiet Punkt download and retry. `main()` alone applies #8's current rule: use an existing `EASY_RAG_VAULT_SOURCE_DIR`, otherwise open the existing picker; cancel means return without writing. `convert_html_to_json()` never opens UI and propagates storage/source/tokenizer errors with a nonzero CLI exit.

## TDD test matrix

Replace `tests/test_vault_store.py` first. Tests use only `tmp_path`, `monkeypatch`, stdlib, and installed project dependencies:

| Test ID | Exact proof |
|---|---|
| `test_load_missing_and_zero_byte_return_empty` | Missing and zero-byte paths both return `[]`; neither creates a backup. |
| `test_unchanged_rerun_keeps_one_entry` | Merge then write/read twice with one normalized identity and identical mtime/hash; result length is one and entry is unchanged. |
| `test_stable_identity_change_replaces_once` | Native-separator `.`/`..` alias plus case alias of one temp path, unchanged mtime, changed 64-hex `content_hash`; one entry remains, incoming chunks/hash win, original position remains. |
| `test_legacy_hash_upgrade_and_no_downgrade` | Missing hash upgrades from incoming hash; later same-mtime incoming legacy entry cannot erase it. |
| `test_corrupt_input_backup_is_unique_and_byte_exact` | Malformed JSON and valid non-list schema each move to separate regex-matching sibling backups; original bytes match exactly and no backup is overwritten. |
| `test_partial_serialization_failure_preserves_target_and_cleans_temp` | Monkeypatched `json.dump` writes partial text then raises; old target bytes remain and `.<name>.*.tmp` glob is empty. |
| `test_replace_failure_preserves_target_and_cleans_temp` | Monkeypatched `os.replace` raises only for temp-to-target replacement; old target bytes remain and temp glob is empty. |
| `test_atomic_output_is_deterministic_and_preserves_mode` | Exact UTF-8 bytes show sorted keys, unescaped Unicode, LF-only lines, two-space indent, one trailing LF; existing POSIX mode bits are retained where `os.chmod` is meaningful. |
| `test_builder_rerun_and_same_mtime_content_change` | Tiny temp HTML source and explicit temp `vault_path`; serial pool/tokenizer seams are monkeypatched, run twice unchanged then change cleaned text while restoring mtime; JSON remains one entry and hash/chunks replace once. |
| `test_builder_import_is_side_effect_free` | Isolated subprocess import in empty temp cwd with download/Tk/pool/file-write sentinels; exit 0, no sentinel call, and no vault artifact. |
| `test_owned_writer_has_no_append_mode` | AST inspection of both owned production files finds no `open`/`Path.open`/`NamedTemporaryFile` mode containing `a` and exactly one `os.replace` call in `atomic_write_json`. |

Tests must not assert a case-folded identity on POSIX; monkeypatch `vault_store.os.path.normcase` to lowercase only for the case-alias assertion, while native `.`/`..` normalization exercises the real host path functions. The builder test must not use the real network, Tk, multiprocessing, or user folders.

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

RED is mandatory: exit 1 with failures for changed-hash replacement, partial-write cleanup, replace cleanup, deterministic bytes, builder explicit output/content-hash behavior, and import-side-effect freedom. Existing missing/valid-write/invalid-backup behavior may pass. If all new tests pass before production edits, stop: the test does not prove the missing behavior. Commit tests only after recording RED:

```text
test(#15): specify atomic vault behavior
```

Implement only the locked production changes, then run:

```powershell
python -m pytest -p no:cacheprovider tests/test_vault_store.py -q
python -m py_compile Vault_json_creation_from_HTMLs.py vault_store.py
rg -n 'mode\s*=\s*["'']a|open\([^)]*,\s*["'']a' Vault_json_creation_from_HTMLs.py vault_store.py
git diff --check
```

GREEN requires 11 focused tests passed, compile and diff checks exit 0, and the append grep returns exit 1 with no matches. Commit production only after GREEN:

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

The focused suite has 11 passes. The full suite has no new failure ID or classification compared with pristine; a pre-existing missing-PyQt5 collection error may remain. Generated `vault.json`, invalid backups, temp files, `__pycache__`, and pytest cache must not be staged or included in closure evidence.

## Threat and rollback boundaries

The trust boundary is local disk content into `load_vault`; schema validation prevents malformed persisted data from silently replacing a valid array. Same-directory temp plus fsync plus replace mitigates partial writes. Backup names contain no source content, and warnings expose paths only to the local console. There is no new package/supply-chain surface and no secret or network input added. The runtime Punkt fallback is pre-existing behavior moved out of import time.

Rollback is reverse-order `git revert` of accepted review-fix commits, then the production commit, then the test commit. Do not restore append mode. Before rollback, preserve any user vault outside Git; the repository rollback never deletes generated data or corruption backups. If the production commit is reverted while the builder remains maintained, restore the last known-good pre-change vault manually from the operator's backup and stop using the builder until the atomic fix is restored.

## Review, activation, commits, and closure

The sequential lifecycle is exactly: **ChatGPT review -> GSD checker -> corrector -> docs merge -> activation -> implementer -> reviewer -> fixer -> verifier -> orchestrator merge/close**.

1. ChatGPT reviews this exact docs head and posts SHA-bound PASS or numbered blockers.
2. GSD checker independently reviews the same head and posts SHA-bound PASS or numbered blockers.
3. The corrector changes only the three packet-owned docs, records every disposition and correction SHA, and repeats both reviews until both PASS the same final head.
4. The orchestrator merges the docs PR and records its immutable merge SHA. Planning remains non-active until a later activation PR/comment names the reviewed packet, frozen GUI SHA, implementation branch, and worker ownership.
5. The implementer records pristine and RED evidence, commits tests, implements, records GREEN/comparison evidence, and commits production. No squash between the two TDD commits before review.
6. The code reviewer reviews the initial production head. The fixer makes one atomic `fix(#15): address accepted review finding` commit per accepted finding; rejected findings get a written reason.
7. The verifier runs every packet command on the final code SHA and reports PASS/BLOCKED without editing code.
8. The orchestrator alone merges the code PR, posts reciprocal evidence links to the issue/canonical packet, updates the roadmap, and closes #15 only when every closure item below exists.

Approval stops are blocking: no docs merge without exact-head ChatGPT and GSD PASS; no code work without the later explicit activation; no code merge/issue closure without final reviewer dispositions and verifier PASS. Planning docs are never implementation evidence.

Closure evidence names the GitHub issue, canonical plan, packet, reviewed docs SHA and merge SHA, activation URL, frozen GUI SHA, test commit, initial production commit, every accepted-finding fix SHA or `none accepted`, final verifier SHA, code PR/merge SHA, pristine/RED/GREEN/full-suite command outputs, append grep, `git diff --check`, and clean generated-data status. It explicitly proves unchanged rerun, stable-identity content change, byte-exact corrupt backup, partial-write and replace-failure target preservation/temp cleanup, builder integration, side-effect-free import, deterministic serialization, and no append mode. It states that #8 remains open and receives no closure credit from #15.
