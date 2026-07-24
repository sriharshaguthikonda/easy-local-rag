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


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConversationImportError("Import contains duplicate object keys.")
        result[key] = value
    return result


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
            object_pairs_hook=_reject_duplicate_keys,
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
