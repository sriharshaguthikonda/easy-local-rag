import hashlib

from token_budget import count_rendered, oldest_turn_removed


def test_count_message_tokens_returns_positive_count(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    spec = {
        "provider": "groq", "model": "fixture", "revision": "v1",
        "tokenizer_path": str(path), "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "max_added_tokens": 0, "add_special_tokens": False,
    }
    count = count_rendered(spec, "hello world", provider="groq", model="fixture", template="fixture-v1")
    assert (count.tokens, count.quality) == (2, "exact")


def test_trim_messages_to_budget_removes_old_turns():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a " * 2000},
        {"role": "assistant", "content": "b " * 2000},
        {"role": "user", "content": "c " * 2000},
    ]
    assert oldest_turn_removed(messages) == [messages[0], messages[-1]]
    assert oldest_turn_removed([messages[0], messages[2], messages[-1]]) is None
