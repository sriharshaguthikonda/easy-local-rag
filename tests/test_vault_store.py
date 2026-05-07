import json
from pathlib import Path

from vault_store import atomic_write_json, load_vault, merge_vault_entries


def test_load_vault_missing_returns_empty(tmp_path):
    assert load_vault(tmp_path / "vault.json") == []


def test_merge_vault_entries_replaces_changed_mod_time():
    existing = [
        {"file_name": "a.html", "modification_time": 10, "chunks": ["old"]},
        {"file_name": "b.html", "modification_time": 20, "chunks": ["b"]},
    ]
    incoming = [
        {"file_name": "a.html", "modification_time": 11, "chunks": ["new"]},
        {"file_name": "c.html", "modification_time": 30, "chunks": ["c"]},
    ]
    merged = merge_vault_entries(existing, incoming)
    by_name = {entry["file_name"]: entry for entry in merged}

    assert len(merged) == 3
    assert by_name["a.html"]["modification_time"] == 11
    assert by_name["a.html"]["chunks"] == ["new"]
    assert by_name["b.html"]["modification_time"] == 20
    assert by_name["c.html"]["modification_time"] == 30


def test_atomic_write_json_writes_valid_json(tmp_path):
    vault = tmp_path / "vault.json"
    payload = [{"file_name": "a.html", "modification_time": 10}]
    atomic_write_json(vault, payload)

    with vault.open("r", encoding="utf-8") as handle:
        assert json.load(handle) == payload


def test_load_vault_backs_up_invalid_json(tmp_path):
    vault = tmp_path / "vault.json"
    vault.write_text("not-json", encoding="utf-8")

    loaded = load_vault(vault)
    backups = list(tmp_path.glob("vault.json.invalid.*.bak"))

    assert loaded == []
    assert not vault.exists()
    assert len(backups) == 1
