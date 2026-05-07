from conversation_import import sanitize_conversation_import, validate_message


def test_validate_message_accepts_valid_message():
    ok, reason = validate_message({"role": "user", "content": "hello"})
    assert ok is True
    assert reason == ""


def test_validate_message_rejects_invalid_role():
    ok, reason = validate_message({"role": "tool", "content": "x"})
    assert ok is False
    assert "Invalid role" in reason


def test_sanitize_conversation_import_ignores_unsafe_keys():
    payload = {
        "history": [{"role": "user", "content": "hello"}],
        "collection": "owned",
        "tts_worker": {"bad": True},
    }
    safe_data, warnings = sanitize_conversation_import(payload)

    assert "history" in safe_data
    assert safe_data["history"] == [{"role": "user", "content": "hello"}]
    assert any("Ignored unsupported keys" in message for message in warnings)
    assert "collection" not in safe_data
    assert "tts_worker" not in safe_data


def test_sanitize_conversation_import_filters_bad_history():
    payload = {
        "history": [
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "bad"},
            {"role": "user", "content": 123},
        ]
    }
    safe_data, warnings = sanitize_conversation_import(payload)

    assert safe_data["history"] == [{"role": "assistant", "content": "ok"}]
    assert len(warnings) >= 2
