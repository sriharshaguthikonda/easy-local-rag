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
