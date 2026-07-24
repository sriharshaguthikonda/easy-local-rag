# Issue #9 Conversation Import Validation Implementation Packet

**Goal:** validate legacy conversation uploads before any Streamlit session write.

[GitHub Issue #9](https://github.com/sriharshaguthikonda/easy-local-rag/issues/9) · [canonical plan](../../issues/ISSUE-009-validate-conversation-import.md) · [roadmap](../../issues/README.md)

**Docs packet lineage:** `codex/issue-009-plan`, based on `8497efca7aa021bef3757c6b87ba8f4a7824fbc5`. Git history carries the packet commits; the immutable final docs-merge SHA is required closure evidence.

**Code lineage:** create `codex/fix-issue-9` directly at frozen GUI base `daecce8a27f50da39284f5519d77b835905209f6`. It targets `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs`; never merge or rebase main/docs into this lineage.

**Owned code files:** `conversation_import.py`, `streamlit_app.py`, `tests/test_conversation_import.py`. Preserve the exporter at `streamlit_app.py:609-620` unchanged.

## Frozen-base caller map

Derived at `daecce8a27f50da39284f5519d77b835905209f6` with this exact command:

```powershell
git grep -n -E "conversation_import|sanitize_conversation_import|validate_message|Import Conversation|uploaded_file" daecce8a27f50da39284f5519d77b835905209f6 -- '*.py'
```

- `conversation_import.py:9` defines legacy `validate_message(message: Any)`; `conversation_import.py:26` defines `sanitize_conversation_import(data: Any)`, whose internal call is at line 42.
- `streamlit_app.py:72` imports `sanitize_conversation_import`; its export/import UI begins at line 607, creates `uploaded_file` at line 622, checks it at line 623, uses `json.loads(uploaded_file.read())` at line 625, then invokes the sanitizer at line 626.
- `tests/test_conversation_import.py:1` imports both names, and its tests at lines 4 and 10 are the only external `validate_message` callers.

No other Python callers were found by that exact `git grep` command. The replacement retires `validate_message`, replaces its only test caller by replacing the whole test file, and changes the sole UI call to the byte-only adapter.

## Locked behavior

`sanitize_conversation_import` accepts **raw `bytes` only**. It rejects more than 8 MiB (`8 * 1024 * 1024`) before UTF-8 decode or JSON parse, then accepts only strict UTF-8 JSON whose root is an object. JSON container depth is at most 32, counting the root object as depth 1 and every nested dict/list as one additional depth. The only top-level keys are `history`, `tags`, `favorites`, and legacy `sources`. `sources` has any JSON value, is ignored, and produces exactly `Ignored legacy export-only key: sources.` once. Every other key, including an underscore-prefixed key, rejects the entire upload.

| Field | Accepted shape and limits | Destination |
|---|---|---|
| `history` | list of at most 1,000 dicts, exactly `role`/`content`; roles only `system`, `user`, `assistant`; string content at most 64 KiB measured in UTF-8 bytes | `conversation_history` |
| `tags` | dict with at most 100 string keys and **at most 100 string values total across all keys**; values are lists of strings; every key and value at most 256 characters | `tags` |
| `favorites` | list of at most 100 dicts; all keys are strings, all values are scalar `str`, `int`, finite `float`, `bool`, or `null`; every string key/value at most 256 characters | `favorite_responses` |

Absent `history`, `tags`, or `favorites` causes no write for that destination. JSON integers are limited to 4,096 decimal digits, excluding a leading minus. Every decoded dict key and string value must strictly UTF-8 encode: lone surrogates reject globally, including under ignored `sources`. `bool` is accepted only as a favorite scalar. `NaN`, infinities, and numeric-overflow literals such as `1e999` reject globally, including under ignored `sources`. Every accepted container is newly built. A failed validation makes zero writes; import data never assigns `sources`, `current_sources`, `collection`, `chroma_client`, `source_filters`, TTS objects, underscore keys, or any runtime key. Importing these modules must not initialize Chroma or Streamlit in tests.

Errors are deterministic: raw type/size, decode/JSON failure, non-object root, unsupported key, and field violations each raise `ConversationImportError` with the exact messages in the implementation below. The UI renders only `Conversation import rejected: {error}`. This makes rejected-input assertions stable and rollback a bounded reverse-order revert of the implementation and any accepted review-fix commits.

## Frozen-base baseline — before edits

In the clean code worktree at `daecce8a27f50da39284f5519d77b835905209f6`, run and record literal command, exit code, test IDs for every failure, and full summary line:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest tests -q
python -m py_compile conversation_import.py streamlit_app.py rag_gui.py GUI_direct_search.py
gh issue list --repo sriharshaguthikonda/easy-local-rag --state open
```

Do not assume PyQt5 or closed-#14 failures: record them only when observed. After the implementation, run the same commands and capture the same fields. The new focused test IDs must pass; every pre-existing failure must have the same test ID and classification, and no new failure is allowed. The `gh` output is saved as live-state evidence, not guessed from this packet.

## Exact test file (write first)

Replace `tests/test_conversation_import.py` with this complete file:

```python
import ast
import json
import re
from pathlib import Path

import pytest

from conversation_import import (
    MAX_FAVORITES, MAX_HISTORY_ENTRIES, MAX_IMPORT_BYTES, MAX_INTEGER_DIGITS, MAX_JSON_DEPTH,
    MAX_MESSAGE_BYTES, MAX_TAGS, MAX_TEXT_CHARS, ConversationImportError,
    apply_conversation_import, apply_uploaded_conversation,
    sanitize_conversation_import,
)


class RecordingState(dict[str, object]):
    def __init__(self, initial: dict[str, object]) -> None:
        super().__init__(initial)
        self.assignments: list[str] = []

    def __setitem__(self, key: str, value: object) -> None:
        self.assignments.append(key)
        super().__setitem__(key, value)


def sentinel_state() -> RecordingState:
    return RecordingState({
        "collection": object(), "chroma_client": object(),
        "current_sources": object(), "source_filters": object(),
        "tts_queue": object(), "tts_worker": object(),
        "conversation_history": ["old-history"], "tags": {"old": ["tag"]},
        "favorite_responses": [{"old": "favorite"}],
    })


def raw(payload: object) -> bytes:
    return json.dumps(payload, allow_nan=False).encode("utf-8")


def valid_payload() -> dict[str, object]:
    return {
        "history": [{"role": "user", "content": "hello"}],
        "tags": {"notes.md": ["keep"]},
        "favorites": [{"file_name": "notes.md", "score": 0.9}],
    }


def test_valid_import_assigns_only_conversation_fields_and_preserves_runtime_sentinels() -> None:
    state = sentinel_state()
    collection, chroma_client = state["collection"], state["chroma_client"]
    current_sources, source_filters = state["current_sources"], state["source_filters"]
    tts_queue, tts_worker = state["tts_queue"], state["tts_worker"]
    state.assignments.clear()
    warnings = apply_conversation_import(state, raw(valid_payload()))
    assert warnings == []
    assert state.assignments == ["conversation_history", "tags", "favorite_responses"]
    assert state["collection"] is collection
    assert state["chroma_client"] is chroma_client
    assert state["current_sources"] is current_sources
    assert state["source_filters"] is source_filters
    assert state["tts_queue"] is tts_queue
    assert state["tts_worker"] is tts_worker


def test_legacy_sources_are_warned_and_never_applied() -> None:
    payload = valid_payload() | {"sources": [{"file_path": "ignored"}]}
    safe, warnings = sanitize_conversation_import(raw(payload))
    assert "sources" not in safe
    assert warnings == ["Ignored legacy export-only key: sources."]


@pytest.mark.parametrize(("payload", "message"), [
    ({"current_sources": [{"file_path": "crafted"}]}, "Unsupported import key."),
    ({"collection": "crafted"}, "Unsupported import key."),
    ({"chroma_client": "crafted"}, "Unsupported import key."),
    ({"source_filters": "crafted"}, "Unsupported import key."),
    ({"tts_queue": "crafted"}, "Unsupported import key."),
    ({"tts_worker": "crafted"}, "Unsupported import key."),
    ({"_runtime": "crafted"}, "Unsupported import key."),
    ({"unexpected": "crafted"}, "Unsupported import key."),
    ({"history": [{"role": "tool", "content": "crafted"}]}, "Message role is invalid."),
    ({"history": [{"role": "user", "content": 1}]}, "Message content must be a string."),
    ({"history": [{"role": "user", "content": "x", "extra": "x"}]}, "Message must contain exactly role and content."),
    ({"tags": {"notes.md": ["ok", 1]}}, "Tag values must be string lists."),
    ({"favorites": [{"file_name": {"nested": "object"}}]}, "Favorite value must be a finite scalar."),
])
def test_rejected_payload_performs_zero_assignments(payload: dict[str, object], message: str) -> None:
    state = sentinel_state()
    before = dict(state)
    state.assignments.clear()
    with pytest.raises(ConversationImportError, match="^" + re.escape(message) + "$"):
        apply_conversation_import(state, raw(payload))
    assert state.assignments == []
    assert dict(state) == before


def test_exact_raw_8_mib_passes() -> None:
    raw_exact = b'{"sources":null}' + b" " * (MAX_IMPORT_BYTES - len(b'{"sources":null}'))
    assert len(raw_exact) == MAX_IMPORT_BYTES
    assert sanitize_conversation_import(raw_exact) == ({}, ["Ignored legacy export-only key: sources."])


def test_exact_message_64_kib_passes() -> None:
    content = "é" * (MAX_MESSAGE_BYTES // len("é".encode("utf-8")))
    assert len(content.encode("utf-8")) == MAX_MESSAGE_BYTES
    safe, warnings = sanitize_conversation_import(raw({"history": [{"role": "user", "content": content}]}))
    assert warnings == []
    assert safe["history"][0]["content"] == content


def test_exact_other_limits_pass() -> None:
    tags = {f"k{i}": ["v"] for i in range(MAX_TAGS)}
    favorites = [{"file_name": "x"} for _ in range(MAX_FAVORITES)]
    history = [{"role": "user", "content": ""} for _ in range(MAX_HISTORY_ENTRIES)]
    safe, warnings = sanitize_conversation_import(raw({"history": history, "tags": tags, "favorites": favorites}))
    assert warnings == []
    assert len(safe["history"]) == MAX_HISTORY_ENTRIES
    assert len(safe["tags"]) == MAX_TAGS
    assert sum(map(len, safe["tags"].values())) == MAX_TAGS
    assert len(safe["favorites"]) == MAX_FAVORITES
    assert sanitize_conversation_import(raw({"tags": {"k" * MAX_TEXT_CHARS: ["v" * MAX_TEXT_CHARS]}, "favorites": [{"k" * MAX_TEXT_CHARS: "v" * MAX_TEXT_CHARS}]}))[1] == []


def test_non_bytes_rejects() -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("Import must be raw bytes.") + "$"):
        sanitize_conversation_import("{}")  # type: ignore[arg-type]


def test_raw_bytes_over_limit_rejects_before_decode() -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("Import exceeds 8 MiB limit.") + "$"):
        sanitize_conversation_import(b"\xff" * (MAX_IMPORT_BYTES + 1))


def test_history_entry_over_limit_rejects() -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("History exceeds 1000 entries.") + "$"):
        sanitize_conversation_import(raw({"history": [{"role": "user", "content": ""}] * (MAX_HISTORY_ENTRIES + 1)}))


def test_message_utf8_byte_over_limit_rejects() -> None:
    content = "é" * (MAX_MESSAGE_BYTES // len("é".encode("utf-8")) + 1)
    assert len(content) < MAX_MESSAGE_BYTES
    assert len(content.encode("utf-8")) > MAX_MESSAGE_BYTES
    with pytest.raises(ConversationImportError, match="^" + re.escape("Message content exceeds 64 KiB.") + "$"):
        sanitize_conversation_import(raw({"history": [{"role": "user", "content": content}]}))


def test_tag_count_and_value_count_over_limits_reject() -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("Tags exceed 100 keys.") + "$"):
        sanitize_conversation_import(raw({"tags": {str(i): [] for i in range(MAX_TAGS + 1)}}))
    with pytest.raises(ConversationImportError, match="^" + re.escape("Tags exceed 100 total values.") + "$"):
        sanitize_conversation_import(raw({"tags": {"k": ["x"] * (MAX_TAGS + 1)}}))


def test_favorite_count_over_limit_rejects() -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("Favorites exceed 100 entries.") + "$"):
        sanitize_conversation_import(raw({"favorites": [{}] * (MAX_FAVORITES + 1)}))


def test_tag_and_favorite_text_over_limit_reject() -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("Tag text exceeds 256 characters.") + "$"):
        sanitize_conversation_import(raw({"tags": {"x" * (MAX_TEXT_CHARS + 1): []}}))
    with pytest.raises(ConversationImportError, match="^" + re.escape("Favorite text exceeds 256 characters.") + "$"):
        sanitize_conversation_import(raw({"favorites": [{"x" * (MAX_TEXT_CHARS + 1): "ok"}]}))
    with pytest.raises(ConversationImportError, match="^" + re.escape("Tag text exceeds 256 characters.") + "$"):
        sanitize_conversation_import(raw({"tags": {"ok": ["x" * (MAX_TEXT_CHARS + 1)]}}))
    with pytest.raises(ConversationImportError, match="^" + re.escape("Favorite text exceeds 256 characters.") + "$"):
        sanitize_conversation_import(raw({"favorites": [{"ok": "x" * (MAX_TEXT_CHARS + 1)}]}))


@pytest.mark.parametrize("raw_bytes, message", [
    (b"\xff", "Import must be UTF-8 JSON bytes."),
    (b"{", "Import must be UTF-8 JSON bytes."),
    (b"[]", "Import root must be a JSON object."),
])
def test_invalid_utf8_malformed_json_and_array_root_reject(raw_bytes: bytes, message: str) -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape(message) + "$"):
        sanitize_conversation_import(raw_bytes)


@pytest.mark.parametrize("raw_bytes", [
    b'{"sources":NaN}', b'{"sources":Infinity}', b'{"sources":-Infinity}',
    b'{"sources":1e999}', b'{"favorites":[{"score":1e999}]}',
])
def test_non_finite_json_numbers_reject_even_in_ignored_sources(raw_bytes: bytes) -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("Import contains a non-finite number.") + "$"):
        sanitize_conversation_import(raw_bytes)


def test_finite_favorite_float_passes() -> None:
    safe, warnings = sanitize_conversation_import(b'{"favorites":[{"score":1.25}]}')
    assert warnings == []
    assert safe == {"favorites": [{"score": 1.25}]}


@pytest.mark.parametrize(("sign", "negative"), [(b"", False), (b"-", True)])
def test_exact_integer_digit_limit_passes(sign: bytes, negative: bool) -> None:
    raw_bytes = b'{"favorites":[{"score":' + sign + b"9" * MAX_INTEGER_DIGITS + b'}]}'
    safe, warnings = sanitize_conversation_import(raw_bytes)
    assert warnings == []
    score = safe["favorites"][0]["score"]
    assert isinstance(score, int) and (score < 0) is negative and abs(score).bit_length() > 0


@pytest.mark.parametrize("raw_bytes", [
    b'{"sources":' + b"9" * (MAX_INTEGER_DIGITS + 1) + b'}',
    b'{"favorites":[{"score":' + b"9" * (MAX_INTEGER_DIGITS + 1) + b'}]}',
    b'{"favorites":[{"score":-' + b"9" * (MAX_INTEGER_DIGITS + 1) + b'}]}',
])
def test_over_cap_integer_rejects_everywhere(raw_bytes: bytes) -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("Import integer exceeds 4096 digits.") + "$"):
        sanitize_conversation_import(raw_bytes)


def test_uploaded_handler_over_cap_integer_makes_zero_writes() -> None:
    class UploadedFile:
        def getvalue(self) -> bytes:
            return b'{"favorites":[{"score":' + b"9" * (MAX_INTEGER_DIGITS + 1) + b'}]}'

    state = sentinel_state()
    state.assignments.clear()
    warnings: list[str] = []
    successes: list[str] = []
    apply_uploaded_conversation(UploadedFile(), state, warnings.append, successes.append)
    assert state.assignments == []
    assert warnings == ["Conversation import rejected: Import integer exceeds 4096 digits."]
    assert successes == []


@pytest.mark.parametrize("template", [
    b'{"history":[{"role":"user","content":"%s"}]}',
    b'{"tags":{"%s":["ok"]}}',
    b'{"tags":{"ok":["%s"]}}',
    b'{"favorites":[{"%s":"ok"}]}',
    b'{"favorites":[{"ok":"%s"}]}',
    b'{"sources":{"nested":["%s"]}}',
])
@pytest.mark.parametrize("escaped", [b"\\ud800", b"\\udc00"])
def test_lone_surrogates_reject_every_payload_string(template: bytes, escaped: bytes) -> None:
    with pytest.raises(ConversationImportError, match="^" + re.escape("Import strings must be valid UTF-8.") + "$"):
        sanitize_conversation_import(template % escaped)


@pytest.mark.parametrize("template", [
    b'{"history":[{"role":"user","content":"%s"}]}',
    b'{"tags":{"%s":["ok"]}}',
    b'{"tags":{"ok":["%s"]}}',
    b'{"favorites":[{"%s":"ok"}]}',
    b'{"favorites":[{"ok":"%s"}]}',
    b'{"sources":{"nested":["%s"]}}',
])
@pytest.mark.parametrize("escaped", [b"\\ud800", b"\\udc00"])
def test_uploaded_lone_surrogates_make_zero_writes(template: bytes, escaped: bytes) -> None:
    class UploadedFile:
        def getvalue(self) -> bytes:
            return template % escaped

    state = sentinel_state()
    state.assignments.clear()
    warnings: list[str] = []
    successes: list[str] = []
    apply_uploaded_conversation(UploadedFile(), state, warnings.append, successes.append)
    assert state.assignments == []
    assert warnings == ["Conversation import rejected: Import strings must be valid UTF-8."]
    assert successes == []


def test_json_depth_limit_and_parser_recursion_reject() -> None:
    nested: object = None
    for _ in range(MAX_JSON_DEPTH - 1):
        nested = {"nested": nested}
    assert sanitize_conversation_import(raw({"sources": nested})) == ({}, ["Ignored legacy export-only key: sources."])
    with pytest.raises(ConversationImportError, match="^" + re.escape("Import nesting exceeds 32 containers.") + "$"):
        sanitize_conversation_import(raw({"sources": {"nested": nested}}))
    very_deep = b'{"sources":' + b"[" * 2_000 + b"null" + b"]" * 2_000 + b"}"
    with pytest.raises(ConversationImportError, match="^" + re.escape("Import nesting exceeds 32 containers.") + "$"):
        sanitize_conversation_import(very_deep)


def test_streamlit_uploader_wiring_ast() -> None:
    module = ast.parse(Path("streamlit_app.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(module) if isinstance(node, ast.ImportFrom) and node.module == "conversation_import"]
    assert len(imports) == 1
    assert [(alias.name, alias.asname) for alias in imports[0].names] == [("apply_uploaded_conversation", None)]
    upload_ifs = [node for node in ast.walk(module) if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "uploaded_file"]
    assert len(upload_ifs) == 1
    body = upload_ifs[0].body
    assert len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call)
    call = body[0].value
    assert isinstance(call.func, ast.Name) and call.func.id == "apply_uploaded_conversation"
    assert call.keywords == []
    assert len(call.args) == 4
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "uploaded_file"
    assert [ast.unparse(arg) for arg in call.args[1:]] == ["st.session_state", "st.warning", "st.success"]


def test_uploaded_handler_reads_bytes_applies_valid_data_and_warns_rejection() -> None:
    class UploadedFile:
        def __init__(self, raw_bytes: bytes) -> None:
            self.raw, self.calls = raw_bytes, 0
        def getvalue(self) -> bytes:
            self.calls += 1
            return self.raw

    valid_state, invalid_state = sentinel_state(), sentinel_state()
    collection, chroma_client = valid_state["collection"], valid_state["chroma_client"]
    current_sources, source_filters = valid_state["current_sources"], valid_state["source_filters"]
    tts_queue, tts_worker = valid_state["tts_queue"], valid_state["tts_worker"]
    valid_state.assignments.clear()
    warnings: list[str] = []
    successes: list[str] = []
    upload = UploadedFile(raw(valid_payload()))
    apply_uploaded_conversation(upload, valid_state, warnings.append, successes.append)
    assert upload.calls == 1
    assert valid_state["collection"] is collection
    assert valid_state["chroma_client"] is chroma_client
    assert valid_state["current_sources"] is current_sources
    assert valid_state["source_filters"] is source_filters
    assert valid_state["tts_queue"] is tts_queue
    assert valid_state["tts_worker"] is tts_worker
    assert valid_state.assignments == ["conversation_history", "tags", "favorite_responses"]
    assert warnings == []
    assert successes == ["Conversation import complete."]
    invalid_state.assignments.clear()
    invalid_warnings: list[str] = []
    invalid_successes: list[str] = []
    apply_uploaded_conversation(UploadedFile(b"{"), invalid_state, invalid_warnings.append, invalid_successes.append)
    assert invalid_state.assignments == []
    assert invalid_warnings[0].startswith("Conversation import rejected:")
    assert invalid_successes == []
```

First record the pristine frozen baseline above. Then, while code remains at frozen `daecce8a27f50da39284f5519d77b835905209f6`, replace **only** `tests/test_conversation_import.py` with this test file and run the red evidence:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest tests/test_conversation_import.py -q
```

Record literal outcome: exit code `2`, collection `ImportError: cannot import name 'MAX_FAVORITES' from 'conversation_import'`; frozen code lacks the new constants, `ConversationImportError`, and both apply helpers. Record the actual command output alongside this expected red result.

## Locked implementation

Replace `conversation_import.py` with:

```python
import json
import math
from collections.abc import Callable, MutableMapping
from typing import Any

MAX_IMPORT_BYTES = 8 * 1024 * 1024
MAX_INTEGER_DIGITS = 4_096
MAX_JSON_DEPTH = 32
MAX_HISTORY_ENTRIES = 1_000
MAX_MESSAGE_BYTES = 64 * 1024
MAX_TAGS = 100
MAX_FAVORITES = 100
MAX_TEXT_CHARS = 256
_TOP_LEVEL_KEYS = {"history", "tags", "favorites", "sources"}
_ROLES = {"system", "user", "assistant"}


class ConversationImportError(ValueError):
    """Raised when an uploaded conversation violates the fixed import contract."""


def _text(value: Any, message: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_TEXT_CHARS:
        raise ConversationImportError(message)
    return value


def _reject_non_finite(value: str) -> None:
    raise ConversationImportError("Import contains a non-finite number.")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ConversationImportError("Import contains a non-finite number.")
    return parsed


def _parse_integer(value: str) -> int:
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    if not digits or len(digits) > MAX_INTEGER_DIGITS:
        raise ConversationImportError("Import integer exceeds 4096 digits.")
    result = 0
    for digit in digits:
        if digit < "0" or digit > "9":
            raise ConversationImportError("Import integer exceeds 4096 digits.")
        result = result * 10 + ord(digit) - ord("0")
    return -result if negative else result


def _validate_json_depth(payload: object) -> None:
    stack: list[tuple[object, int]] = [(payload, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise ConversationImportError("Import nesting exceeds 32 containers.")
        if isinstance(value, dict):
            stack.extend((child, depth + 1) for child in value.values() if isinstance(child, (dict, list)))
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value if isinstance(child, (dict, list)))


def _validate_strings(payload: object) -> None:
    stack = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ConversationImportError("Import strings must be valid UTF-8.") from error
        elif isinstance(value, dict):
            for key, child in value.items():
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise ConversationImportError("Import strings must be valid UTF-8.") from error
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)


def sanitize_conversation_import(raw: bytes) -> tuple[dict[str, object], list[str]]:
    if not isinstance(raw, bytes):
        raise ConversationImportError("Import must be raw bytes.")
    if len(raw) > MAX_IMPORT_BYTES:
        raise ConversationImportError("Import exceeds 8 MiB limit.")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_non_finite,
            parse_float=_parse_finite_float,
            parse_int=_parse_integer,
        )
    except RecursionError as error:
        raise ConversationImportError("Import nesting exceeds 32 containers.") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversationImportError("Import must be UTF-8 JSON bytes.") from error
    if not isinstance(payload, dict):
        raise ConversationImportError("Import root must be a JSON object.")
    _validate_json_depth(payload)
    _validate_strings(payload)
    if any(not isinstance(key, str) or key.startswith("_") or key not in _TOP_LEVEL_KEYS for key in payload):
        raise ConversationImportError("Unsupported import key.")
    safe: dict[str, object] = {}
    if "history" in payload:
        history = payload["history"]
        if not isinstance(history, list):
            raise ConversationImportError("History must be a list.")
        if len(history) > MAX_HISTORY_ENTRIES:
            raise ConversationImportError("History exceeds 1000 entries.")
        result: list[dict[str, str]] = []
        for message in history:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ConversationImportError("Message must contain exactly role and content.")
            role, content = message["role"], message["content"]
            if not isinstance(role, str) or role not in _ROLES:
                raise ConversationImportError("Message role is invalid.")
            if not isinstance(content, str):
                raise ConversationImportError("Message content must be a string.")
            if len(content.encode("utf-8")) > MAX_MESSAGE_BYTES:
                raise ConversationImportError("Message content exceeds 64 KiB.")
            result.append({"role": role, "content": content})
        safe["history"] = result
    if "tags" in payload:
        tags = payload["tags"]
        if not isinstance(tags, dict):
            raise ConversationImportError("Tags must be an object.")
        if len(tags) > MAX_TAGS:
            raise ConversationImportError("Tags exceed 100 keys.")
        total_values = 0
        result_tags: dict[str, list[str]] = {}
        for key, values in tags.items():
            key = _text(key, "Tag text exceeds 256 characters.")
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ConversationImportError("Tag values must be string lists.")
            total_values += len(values)
            if total_values > MAX_TAGS:
                raise ConversationImportError("Tags exceed 100 total values.")
            result_tags[key] = [_text(value, "Tag text exceeds 256 characters.") for value in values]
        safe["tags"] = result_tags
    if "favorites" in payload:
        favorites = payload["favorites"]
        if not isinstance(favorites, list):
            raise ConversationImportError("Favorites must be a list.")
        if len(favorites) > MAX_FAVORITES:
            raise ConversationImportError("Favorites exceed 100 entries.")
        result_favorites: list[dict[str, str | int | float | bool | None]] = []
        for favorite in favorites:
            if not isinstance(favorite, dict):
                raise ConversationImportError("Favorite must be an object.")
            clean: dict[str, str | int | float | bool | None] = {}
            for key, value in favorite.items():
                key = _text(key, "Favorite text exceeds 256 characters.")
                if isinstance(value, str):
                    clean[key] = _text(value, "Favorite text exceeds 256 characters.")
                elif value is None or isinstance(value, bool) or isinstance(value, int):
                    clean[key] = value
                elif isinstance(value, float) and math.isfinite(value):
                    clean[key] = value
                else:
                    raise ConversationImportError("Favorite value must be a finite scalar.")
            result_favorites.append(clean)
        safe["favorites"] = result_favorites
    return safe, ["Ignored legacy export-only key: sources."] if "sources" in payload else []


def apply_conversation_import(session_state: MutableMapping[str, object], raw: bytes) -> list[str]:
    safe_data, warnings = sanitize_conversation_import(raw)
    if "history" in safe_data:
        session_state["conversation_history"] = safe_data["history"]
    if "tags" in safe_data:
        session_state["tags"] = safe_data["tags"]
    if "favorites" in safe_data:
        session_state["favorite_responses"] = safe_data["favorites"]
    return warnings


def apply_uploaded_conversation(uploaded_file: Any, session_state: MutableMapping[str, object], show_warning: Callable[[str], None], show_success: Callable[[str], None]) -> None:
    try:
        warnings = apply_conversation_import(session_state, uploaded_file.getvalue())
    except ConversationImportError as error:
        show_warning(f"Conversation import rejected: {error}")
        return
    for warning in warnings:
        show_warning(warning)
    show_success("Conversation import complete.")
```

In `streamlit_app.py`, replace the existing `sanitize_conversation_import` import with `from conversation_import import apply_uploaded_conversation`. Replace the uploader `if uploaded_file:` body exactly with:

```python
if uploaded_file:
    apply_uploaded_conversation(uploaded_file, st.session_state, st.warning, st.success)
```

There must be no `json.loads(uploaded_file.read())`, broad exception handler, or direct importer session assignments. The exporter remains byte-for-byte schema-compatible.

## TDD, review, and closure lifecycle

The lifecycle is exactly: **planner packet -> ChatGPT review -> GSD checker -> corrector -> docs merge -> implementer initial TDD commit -> code-review agent -> accepted-finding fixer commit(s) -> verifier -> orchestrator PR merge/evidence/close**. No review is required before code exists; code review occurs after the initial implementation commit.

## ChatGPT review ledger

Historical ChatGPT review of `8d3f8d1e344e607235c362d46a589d313c1ee28d` returned **5 BLOCKERS**: (1) authority links, corrected by `28e3e469c8274a0e9197fe328570b53daaabbd1e`; (2) caller/wiring proof, corrected by `7ed0edfb0f7f918d82f99e6e151a59ee4b1517fd`; (3) multibyte UTF-8 boundary, corrected by `28e3e469c8274a0e9197fe328570b53daaabbd1e`; (4) deep JSON rejection, corrected by `7ed0edfb0f7f918d82f99e6e151a59ee4b1517fd`; and (5) per-head review evidence, corrected by `7ed0edfb0f7f918d82f99e6e151a59ee4b1517fd`.

For head `99e0bc7`, [ChatGPT posted 2 BLOCKERS](https://github.com/sriharshaguthikonda/easy-local-rag/pull/31#issuecomment-5065717085); this commit corrects global surrogate rejection and public review gates. [Historical GSD PASS evidence](https://github.com/sriharshaguthikonda/easy-local-rag/pull/31#issuecomment-5065717191) is historical only, not a verdict for a later head. This is not a final PASS. A fresh review is required for the exact new head; its ledger entry must record reviewed SHA, actual PASS/BLOCKERS, numbered dispositions and correction SHAs, and re-review result. Never copy an older outcome forward or edit this packet after final PASS.

PR comments, not Q&A, are verification authority for this gate. The [standing public authorization](https://github.com/sriharshaguthikonda/easy-local-rag/pull/31#issuecomment-5065716983) is the authorization record. Before docs merge/implementation and again before code-PR merge/closure, the orchestrator verifies current exact-head ChatGPT and GSD PR comment/review URLs that name the SHA, verdict, findings, and dispositions; it also verifies that no later public GitHub comment revokes authorization. If any evidence is missing, blocking, or revoked: **STOP; do not proceed**. Closure cites the comment URLs and IDs.

After docs merge, the implementer writes the test file, records its red collection result, implements the locked code, and makes `fix(#9): validate conversation imports`. Then the code-review agent reviews that commit; each accepted finding gets an atomic `fix(#9): address accepted review finding` commit. The verifier evaluates the final code SHA. Closure evidence names the docs merge SHA, initial code SHA, every review-fix SHA (or `none accepted`), final verifier SHA, code PR URL, exact commands/outcomes, and the frozen/post-change comparison.

Run this identical comparison block after change, with bytecode disabled, and record command, exit code, full summary, and failing test IDs:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pytest tests -q
python -m py_compile conversation_import.py streamlit_app.py rag_gui.py GUI_direct_search.py
gh issue list --repo sriharshaguthikonda/easy-local-rag --state open
```

Run these extra gates outside the identical comparison block:

```powershell
python -m pytest tests/test_conversation_import.py -q
git diff --check
```

Closure additionally records: exact 8 MiB pass; 8 MiB+1 pre-decode rejection; exact 64 KiB pass; 64 KiB+1 rejection; exact and one-over history/tag-key/tag-total-value/favorite/text/integer limits; high/low-surrogate rejection in every payload string location and uploader zero-write proof; exact-depth/depth-33/parser-recursion rejection; malformed UTF-8/JSON/root rejection; non-finite JSON rejection; AST uploader-wiring proof; sentinel identity; valid and invalid uploader calls with fresh state and separate warning/success recorders; and zero assignments after construction assignments are explicitly cleared. Roll back by reverting the initial code commit and any accepted review-fix commits in reverse order.
