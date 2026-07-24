# Issue #9 Conversation Import Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept only bounded, validated legacy conversation data without allowing an imported file to replace trusted Streamlit runtime state.

**Architecture:** `conversation_import.py` is the pure byte-to-whitelisted-state boundary and performs all validation before it mutates a supplied `MutableMapping`. `streamlit_app.py` reads upload bytes, delegates once to that boundary, and renders controlled warnings; it never parses import JSON or writes imported state itself.

**Tech Stack:** Python standard library (`json`, `typing`), Streamlit, pytest.

## Global Constraints

- Mode is maintained path; retain the Streamlit conversation export/import feature.
- Add no dependency and change no exported JSON schema.
- Create `codex/fix-issue-9` directly from frozen GUI SHA `daecce8a27f50da39284f5519d77b835905209f6`; do not merge or rebase main or docs lineage into that GUI lineage.
- This packet is on docs branch `codex/issue-009-plan` at docs base `8497efca7aa021bef3757c6b87ba8f4a7824fbc5`.
- The sole code-owned files are `conversation_import.py`, `streamlit_app.py`, and `tests/test_conversation_import.py`.
- Reject an invalid import atomically: no assignment to the supplied mapping before full validation succeeds.
- Never assign `sources`, `current_sources`, `collection`, `chroma_client`, `source_filters`, TTS objects, underscore-prefixed keys, or any other runtime key from import data.
- Raw bytes are limited to 8 MiB before UTF-8 decoding or JSON parsing; history is limited to 1,000 entries; every message content is limited to 64 KiB measured as UTF-8 bytes; tags and favorites are each limited to 100 entries; every tag/favorite string is limited to 256 characters.
- Roles are exactly `system`, `user`, and `assistant`.

---

## Authority, lineage, and caller map

- [GitHub Issue #9](https://github.com/sriharshaguthikonda/easy-local-rag/issues/9)
- [Stable main canonical plan](../../issues/ISSUE-009-validate-conversation-import.md)
- [Stable main roadmap ledger](../../issues/README.md)
- Docs packet branch/base: `codex/issue-009-plan` / `8497efca7aa021bef3757c6b87ba8f4a7824fbc5`.
- Code branch/base: `codex/fix-issue-9` / `daecce8a27f50da39284f5519d77b835905209f6`, branched directly from `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.
- Packet/docs commit: `docs(#9): add conversation import JIT packet`.

| Current GUI location at frozen lineage | Current behavior | Code-worker action |
|---|---|---|
| `conversation_import.py:4-87`, `validate_message`, `sanitize_conversation_import` | Drops individual invalid values and accepts parsed objects. | Replace with byte validation plus an atomic application helper. |
| `streamlit_app.py:72`, import of `sanitize_conversation_import` | Imports only the permissive sanitizer. | Import the atomic upload adapter. |
| `streamlit_app.py:614-641`, `main` conversation uploader | Calls `json.loads(uploaded_file.read())`, then directly assigns three session values. | Read bytes once, delegate, and show controlled warnings. |
| `streamlit_app.py:484-492`, `main` tag/favorite controls | Tags are `dict[str, list[str]]`; favorites append the current source snapshot dict. | Preserve these legacy shapes; import never restores the separate `sources` export key. |
| `streamlit_app.py:119-145` module session initialization | Owns collection, Chroma client, current sources, filters, and TTS runtime state. | Preserve their identity across every import result. |
| `tests/test_conversation_import.py:1-38` | Covers permissive sanitization only. | Replace with direct boundary and executable uploader-handler tests. |

## Locked import contract

The pre-existing exporter in `streamlit_app.py:609-620` writes the top-level keys `history`, `sources`, `tags`, and `favorites`. `sources` serializes `st.session_state.current_sources`; it is an export-only legacy key. A valid legacy file containing it is accepted with exactly one warning, `Ignored legacy export-only key: sources.`, and it is never returned or assigned. The maintained data mapping is `history` to `conversation_history`, `tags` to `tags`, and `favorites` to `favorite_responses`.

| Input key | Exact accepted legacy shape | Output mapping key | Rejection rule |
|---|---|---|---|
| `history` | `list[dict]`; each dict has exactly `role` and `content`; role is `system`, `user`, or `assistant`; content is `str`. | `conversation_history` | More than 1,000 entries, non-object entry, extra/missing key, unsupported role, non-string content, or content over 65,536 UTF-8 bytes rejects the whole file. |
| `tags` | `dict[str, list[str]]`, matching `tags.split(",")` at `streamlit_app.py:487`. | `tags` | More than 100 keys, more than 100 total strings, non-string key/value, or text over 256 characters rejects the whole file. |
| `favorites` | `list[dict[str, str | int | float | bool | None]]`, matching source snapshot dictionaries appended at `streamlit_app.py:492`; only shallow scalar values are carried. | `favorite_responses` | More than 100 dictionaries, non-string key, nested container, or string key/value over 256 characters rejects the whole file. |
| `sources` | Any JSON value in a legacy export. | none | Omit it and emit the single fixed warning above. |
| absent `history`, `tags`, or `favorites` | absent | none | Do not overwrite that session key. |
| any other key | none | none | Reject the whole file, including underscore-prefixed and runtime-object keys. |

The top-level object must use string keys. Every accepted list and dictionary is rebuilt into fresh containers; `bool` is accepted only as a favorite scalar, not as a role, count, string, or message content. JSON decoding errors, invalid UTF-8, oversized raw bytes, bad root type, and every contract violation raise `ConversationImportError`. No warning is a partial-import path.

## Task 1: Prove the pure atomic import boundary

**Files:**

- Modify: `conversation_import.py:1-87`
- Modify: `tests/test_conversation_import.py:1-38`

**Interfaces:**

```python
from collections.abc import Callable, MutableMapping
from typing import Any

MAX_IMPORT_BYTES = 8 * 1024 * 1024
MAX_HISTORY_ENTRIES = 1_000
MAX_MESSAGE_BYTES = 64 * 1024
MAX_TAGS = 100
MAX_FAVORITES = 100
MAX_TEXT_CHARS = 256

class ConversationImportError(ValueError):
    pass

def sanitize_conversation_import(raw: bytes) -> tuple[dict[str, object], list[str]]:
    pass

def apply_conversation_import(
    session_state: MutableMapping[str, object], raw: bytes
) -> list[str]:
    pass
```

- [ ] **Step 1: Write the failing atomic-boundary tests**

Replace `tests/test_conversation_import.py` with tests importing `json`, `pytest`, and every interface above. Define this exact recording mapping and sentinel factory:

```python
class RecordingState(dict[str, object]):
    def __init__(self, initial: dict[str, object]) -> None:
        super().__init__(initial)
        self.assignments: list[str] = []

    def __setitem__(self, key: str, value: object) -> None:
        self.assignments.append(key)
        super().__setitem__(key, value)

def sentinel_state() -> RecordingState:
    return RecordingState(
        {
            "collection": object(),
            "chroma_client": object(),
            "current_sources": object(),
            "source_filters": object(),
            "tts_queue": object(),
            "tts_worker": object(),
            "conversation_history": ["old-history"],
            "tags": {"old": ["tag"]},
            "favorite_responses": [{"old": "favorite"}],
        }
    )
```

Add `test_valid_import_assigns_only_conversation_fields_and_preserves_runtime_sentinels`. It captures each runtime sentinel by identity, calls `apply_conversation_import` with `json.dumps` encoded UTF-8 data containing one valid history message, `{"notes.md": ["keep"]}` tags, and `[{"file_name": "notes.md", "score": 0.9}]` favorites, then asserts:

```python
assert warnings == []
assert state.assignments == [
    "conversation_history",
    "tags",
    "favorite_responses",
]
assert state["collection"] is collection
assert state["chroma_client"] is chroma_client
assert state["current_sources"] is current_sources
assert state["source_filters"] is source_filters
assert state["tts_queue"] is tts_queue
assert state["tts_worker"] is tts_worker
```

Add `test_legacy_sources_are_warned_and_never_applied` with this payload and assertions:

```python
payload = {
    "history": [{"role": "user", "content": "hello"}],
    "tags": {"notes.md": ["keep"]},
    "favorites": [{"file_name": "notes.md", "score": 0.9}],
    "sources": [{"file_path": "ignored"}],
}
safe, warnings = sanitize_conversation_import(json.dumps(payload).encode("utf-8"))
assert "sources" not in safe
assert warnings == ["Ignored legacy export-only key: sources."]
```

Add `test_rejected_payload_performs_zero_assignments` parametrized with each exact payload below. It snapshots `dict(state)`, clears `state.assignments`, expects `ConversationImportError` from `apply_conversation_import`, then asserts `state.assignments == []` and `dict(state) == before`.

```python
[
    {"current_sources": [{"file_path": "crafted"}]},
    {"collection": "crafted"},
    {"chroma_client": "crafted"},
    {"source_filters": "crafted"},
    {"tts_queue": "crafted"},
    {"tts_worker": "crafted"},
    {"_runtime": "crafted"},
    {"unexpected": "crafted"},
    {"history": [{"role": "tool", "content": "crafted"}]},
    {"history": [{"role": "user", "content": 1}]},
    {"history": [{"role": "user", "content": "x", "extra": "x"}]},
    {"tags": {"notes.md": ["ok", 1]}},
    {"favorites": [{"file_name": {"nested": "object"}}]},
]
```

Add `test_exact_limits_pass` and these six one-over tests: `test_raw_bytes_over_limit_rejects_before_decode`, `test_history_entry_over_limit_rejects`, `test_message_utf8_byte_over_limit_rejects`, `test_tag_count_and_value_count_over_limits_reject`, `test_favorite_count_over_limit_rejects`, and `test_tag_and_favorite_text_over_limit_reject`. The passing payload uses exactly `MAX_HISTORY_ENTRIES` empty-content user messages, exactly `MAX_TAGS` keys with one tag each, exactly `MAX_FAVORITES` `{"file_name": "x"}` dictionaries, and strings exactly `MAX_TEXT_CHARS` characters. The one-over message uses `"x" * (MAX_MESSAGE_BYTES + 1)`, the raw-byte test passes `b"x" * (MAX_IMPORT_BYTES + 1)`, and each other one-over input uses the named constant plus one.

Add `test_invalid_utf8_malformed_json_and_array_root_reject`:

```python
@pytest.mark.parametrize("raw", [b"\xff", b"{", b"[]"])
def test_invalid_utf8_malformed_json_and_array_root_reject(raw: bytes) -> None:
    with pytest.raises(ConversationImportError):
        sanitize_conversation_import(raw)
```

- [ ] **Step 2: Run the focused test to establish red evidence**

Run:

```powershell
python -m pytest tests/test_conversation_import.py -q
```

Expected: FAIL during collection because frozen code does not export `ConversationImportError`, the byte/count constants, or `apply_conversation_import`.

- [ ] **Step 3: Implement the smallest pure boundary**

In `conversation_import.py`, replace the old permissive functions. Import `json`, `MutableMapping` from `collections.abc`, and `Any` from `typing`. Define the constants and exception from the interface block. Make `sanitize_conversation_import(raw)` first reject `not isinstance(raw, bytes)` and `len(raw) > MAX_IMPORT_BYTES`; then call `raw.decode("utf-8")` and `json.loads` inside a `try` that raises `ConversationImportError("Import must be UTF-8 JSON bytes.")` for `UnicodeDecodeError` and `json.JSONDecodeError`.

Validate a root `dict` before reading fields. Permit only `history`, `tags`, `favorites`, and `sources`; reject any non-string key, every key beginning `_`, and every key outside that set with `ConversationImportError("Unsupported import key.")`. Build fresh `list` and `dict` values for every accepted field. Require message keys exactly `{"role", "content"}`, use `len(content.encode("utf-8"))` for the message byte limit, and use `isinstance(value, str)` for all tags. Require favorite dictionaries to have string keys and values in `(str, int, float, bool)` or `None`, reject nested mappings/lists, and check 256-character limits for every favorite string key/value. Return the fixed `sources` warning only when `sources` was present.

Add this complete application function after the sanitizer:

```python
def apply_conversation_import(
    session_state: MutableMapping[str, object], raw: bytes
) -> list[str]:
    safe_data, warnings = sanitize_conversation_import(raw)
    if "history" in safe_data:
        session_state["conversation_history"] = safe_data["history"]
    if "tags" in safe_data:
        session_state["tags"] = safe_data["tags"]
    if "favorites" in safe_data:
        session_state["favorite_responses"] = safe_data["favorites"]
    return warnings
```

- [ ] **Step 4: Prove the pure boundary is green**

Run:

```powershell
python -m pytest tests/test_conversation_import.py -q
python -m py_compile conversation_import.py
```

Expected: focused tests PASS and compilation exits 0.

## Task 2: Execute the real uploader handler without live Chroma

**Files:**

- Modify: `conversation_import.py`
- Modify: `streamlit_app.py:72,614-641`
- Modify: `tests/test_conversation_import.py`

**Interfaces:**

```python
def apply_uploaded_conversation(
    uploaded_file: Any,
    session_state: MutableMapping[str, object],
    show_warning: Callable[[str], None],
    show_success: Callable[[str], None],
) -> None:
    raw_import = uploaded_file.getvalue()
    try:
        import_warnings = apply_conversation_import(session_state, raw_import)
    except ConversationImportError as error:
        show_warning(f"Conversation import rejected: {error}")
        return
    for warning in import_warnings:
        show_warning(warning)
    show_success("Conversation import complete.")
```

- [ ] **Step 1: Add an executable Streamlit handler-boundary test**

Add `test_uploaded_handler_reads_bytes_applies_valid_data_and_warns_rejection` to `tests/test_conversation_import.py`. Import `apply_uploaded_conversation` from `conversation_import`; this maintained test seam does not import Streamlit, Chroma, or a live collection.

Define this local upload fake and callback recorders:

```python
class UploadedFile:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.calls = 0

    def getvalue(self) -> bytes:
        self.calls += 1
        return self.raw

warnings: list[str] = []
successes: list[str] = []
```

Call `apply_uploaded_conversation` with a valid upload, `sentinel_state()`, `warnings.append`, and `successes.append`; assert `upload.calls == 1`, all runtime sentinels remain identical, `warnings == []`, and `successes == ["Conversation import complete."]`. Call it again with `UploadedFile(b"{")`; assert the recorded mapping has no assignments, the warning begins `Conversation import rejected:`, and `successes == []`. This executes the same bytes-to-helper-to-controlled-warning path used by Streamlit and proves it has no Chroma/import side effect.

- [ ] **Step 2: Run the new handler test for red evidence**

Run:

```powershell
python -m pytest tests/test_conversation_import.py::test_uploaded_handler_reads_bytes_applies_valid_data_and_warns_rejection -q
```

Expected: FAIL during import because the frozen code has no upload adapter.

- [ ] **Step 3: Add the upload adapter and replace the Streamlit branch**

Add the `apply_uploaded_conversation` function from the interface block to `conversation_import.py`; import `Callable` and `MutableMapping` from `collections.abc`. At `streamlit_app.py:72`, replace the old import with:

```python
from conversation_import import apply_uploaded_conversation
```

At `streamlit_app.py:624-641`, replace the entire `if uploaded_file:` body with:

```python
if uploaded_file:
    apply_uploaded_conversation(
        uploaded_file,
        st.session_state,
        st.warning,
        st.success,
    )
```

Do not retain `json.loads(uploaded_file.read())`, `sanitize_conversation_import(imported_data)`, a broad `except Exception`, or direct assignments to `st.session_state` in the upload branch. Leave the exporter unchanged.

- [ ] **Step 4: Run focused and compilation evidence**

Run:

```powershell
python -m pytest tests/test_conversation_import.py -q
python -m py_compile conversation_import.py streamlit_app.py
git diff --check
```

Expected: all focused tests PASS, both files compile with exit 0, and `git diff --check` has no output.

- [ ] **Step 5: Record the repository baseline without changing it**

Run in the code worktree, not this docs worktree:

```powershell
python -m pytest tests -q
```

Expected: record the known baseline PyQt5 collection failure and the closed-#14 citation assertion failure if they occur. No new failure is permitted. This docs worktree has no code/test lineage, so it is not a test target.

- [ ] **Step 6: Stop for code-review, fixer, and verifier evidence, then make the single code commit**

Approval stop: packet review and merge must happen before implementation; then code review, fixer, and verifier evidence must approve the focused, compilation, and baseline record before code PR merge. After those approvals, run:

```powershell
git add conversation_import.py streamlit_app.py tests/test_conversation_import.py
git commit -m "fix(#9): validate conversation imports"
```

Expected: exactly one code commit containing only the three code-owned files. The code PR targets `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`.

## Rollback, close gate, and evidence

- Roll back with one revert of `fix(#9): validate conversation imports`; never restore bulk update or a partial-import loop.
- Before closing #9, post evidence linking this canonical plan, the roadmap, the code PR, and the exact code SHA.
- Closure evidence contains the focused test PASS output, `py_compile` PASS output, the full-suite baseline result, the executable sentinel test proof, the rejected malformed/oversize/malicious zero-assignment proof, and confirmation that the legacy `sources` key produced its fixed warning without replacing `current_sources`.
- No issue closure, merge, or GitHub mutation occurs until the packet-review and code-review/fixer/verifier approval stops have passed.
