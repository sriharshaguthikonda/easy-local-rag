from token_budget import count_message_tokens, trim_messages_to_budget


def test_count_message_tokens_returns_positive_count():
    messages = [{"role": "user", "content": "hello world"}]
    assert count_message_tokens(messages, "llama-3.3-70b-versatile") > 0


def test_trim_messages_to_budget_removes_old_turns():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a " * 2000},
        {"role": "assistant", "content": "b " * 2000},
        {"role": "user", "content": "c " * 2000},
    ]
    trimmed, changed = trim_messages_to_budget(
        messages,
        max_input_tokens=200,
        model_name="llama-3.3-70b-versatile",
    )
    assert changed is True
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["role"] == "user"
