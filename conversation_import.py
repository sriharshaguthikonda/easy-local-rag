from typing import Any


ALLOWED_KEYS = {"history", "tags", "favorites"}
ALLOWED_ROLES = {"system", "user", "assistant"}
MAX_MESSAGE_LENGTH = 20000


def validate_message(message: Any) -> tuple[bool, str]:
    if not isinstance(message, dict):
        return False, "Message is not an object."

    role = message.get("role")
    content = message.get("content")

    if role not in ALLOWED_ROLES:
        return False, f"Invalid role: {role!r}."
    if not isinstance(content, str):
        return False, "Message content must be a string."
    if len(content) > MAX_MESSAGE_LENGTH:
        return False, "Message content exceeds size limit."

    return True, ""


def sanitize_conversation_import(data: Any) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    safe_data: dict[str, Any] = {}

    if not isinstance(data, dict):
        return {}, ["Import payload must be a JSON object."]

    unknown_keys = sorted(set(data.keys()) - ALLOWED_KEYS)
    if unknown_keys:
        warnings.append(f"Ignored unsupported keys: {', '.join(unknown_keys)}")

    history = data.get("history", [])
    if history is not None:
        if isinstance(history, list):
            valid_history = []
            for idx, message in enumerate(history):
                ok, reason = validate_message(message)
                if ok:
                    valid_history.append(
                        {"role": message["role"], "content": message["content"]}
                    )
                else:
                    warnings.append(f"Ignored history[{idx}]: {reason}")
            safe_data["history"] = valid_history
        else:
            warnings.append("Ignored history: expected a list.")

    tags = data.get("tags", {})
    if tags is not None:
        if isinstance(tags, dict):
            safe_tags: dict[str, Any] = {}
            for key, value in tags.items():
                if not isinstance(key, str):
                    warnings.append("Ignored a tag key that is not a string.")
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    safe_tags[key] = value
                elif isinstance(value, list) and all(
                    isinstance(item, (str, int, float, bool)) or item is None
                    for item in value
                ):
                    safe_tags[key] = value
                else:
                    warnings.append(f"Ignored tag '{key}': unsupported value type.")
            safe_data["tags"] = safe_tags
        else:
            warnings.append("Ignored tags: expected an object.")

    favorites = data.get("favorites", [])
    if favorites is not None:
        if isinstance(favorites, list):
            safe_favorites = []
            for idx, value in enumerate(favorites):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    safe_favorites.append(value)
                elif isinstance(value, dict):
                    safe_favorites.append(value)
                else:
                    warnings.append(f"Ignored favorites[{idx}]: unsupported type.")
            safe_data["favorites"] = safe_favorites
        else:
            warnings.append("Ignored favorites: expected a list.")

    return safe_data, warnings
