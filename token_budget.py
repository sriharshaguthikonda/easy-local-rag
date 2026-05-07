from typing import Any


def _safe_encode_length(text: str, model_name: str) -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text or ""))
    except Exception:
        words = len((text or "").split())
        return int(words * 1.6)


def count_message_tokens(messages: list[dict[str, Any]], model_name: str) -> int:
    total = 0
    for message in messages:
        total += _safe_encode_length(str(message.get("content", "")), model_name)
        total += 4
    return total


def trim_messages_to_budget(
    messages: list[dict[str, Any]],
    max_input_tokens: int,
    model_name: str,
) -> tuple[list[dict[str, Any]], bool]:
    if not messages:
        return [], False

    trimmed = list(messages)
    changed = False

    # Keep system at index 0 and current user message at the end.
    while (
        count_message_tokens(trimmed, model_name) > max_input_tokens
        and len(trimmed) > 2
    ):
        del trimmed[1]
        changed = True

    if count_message_tokens(trimmed, model_name) > max_input_tokens and len(trimmed) >= 2:
        user_message = dict(trimmed[-1])
        content = str(user_message.get("content", ""))
        while content and count_message_tokens(trimmed, model_name) > max_input_tokens:
            content = content[: int(len(content) * 0.9)]
            user_message["content"] = content
            trimmed[-1] = user_message
            changed = True

    return trimmed, changed
