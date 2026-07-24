import ast
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import vault_store
from vault_store import atomic_write_json, load_vault, merge_vault_entries


HASH_A = "a" * 64
HASH_B = "b" * 64


def _entry(path="a.html", mtime=1, content_hash=HASH_A, chunks=None, **extra):
    value = {
        "file_name": path,
        "modification_time": mtime,
        "chunks": chunks if chunks is not None else [{"id": HASH_A, "text": "text"}],
    }
    if content_hash is not None:
        value["content_hash"] = content_hash
    value.update(extra)
    return value


def _valid_payload():
    return [_entry()]


def _assert_preserved(vault, old, exc_info):
    assert exc_info.value is exc_info.value
    assert vault.read_bytes() == old


def test_load_missing_and_zero_byte_return_empty(tmp_path):
    missing = tmp_path / "missing.json"
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    assert load_vault(missing) == []
    assert load_vault(empty) == []
    assert not list(tmp_path.glob("*.invalid.*.bak"))


def test_unchanged_rerun_keeps_one_entry_and_object_value(tmp_path):
    path = str(tmp_path / "folder" / "." / "child" / ".." / "a.html")
    existing = [_entry(path, 1, HASH_A, marker="existing")]
    incoming = [_entry(path, 1, HASH_A, marker="incoming")]
    before_existing, before_incoming = copy.deepcopy(existing), copy.deepcopy(incoming)
    merged = merge_vault_entries(existing, incoming)
    assert merged == [existing[0]]
    assert merged[0] is existing[0]
    assert existing == before_existing and incoming == before_incoming
    vault = tmp_path / "vault.json"
    atomic_write_json(vault, merged)
    first = vault.read_bytes()
    merged_again = merge_vault_entries(merged, incoming)
    atomic_write_json(vault, merged_again)
    assert len(merged_again) == 1
    assert merged_again[0] is merged[0]
    assert vault.read_bytes() == first


def test_changed_mtime_with_unchanged_or_absent_hash_replaces_once(tmp_path):
    alias = str(tmp_path / "native" / "." / "sub" / ".." / "doc.html")
    for old_hash, new_hash in ((HASH_A, HASH_A), (None, None)):
        existing = [_entry(alias, 1, old_hash, chunks=[{"id": HASH_A, "text": "old"}])]
        incoming = [_entry(alias, 2, new_hash, chunks=[{"id": HASH_B, "text": "new"}])]
        merged = merge_vault_entries(existing, incoming)
        assert len(merged) == 1
        assert merged[0]["modification_time"] == 2
        assert merged[0]["chunks"] == incoming[0]["chunks"]


def test_same_mtime_changed_hash_replaces_once():
    existing = [_entry("a.html", 1, HASH_A, chunks=[{"id": HASH_A, "text": "old"}])]
    incoming = [_entry("a.html", 1, HASH_B, chunks=[{"id": HASH_B, "text": "new"}])]
    merged = merge_vault_entries(existing, incoming)
    assert len(merged) == 1
    assert merged[0]["content_hash"] == HASH_B
    assert merged[0]["chunks"] == incoming[0]["chunks"]


def test_legacy_hash_upgrade_merge_winners_and_no_mutation():
    existing = [_entry("a.html", 1, None, chunks=[{"id": HASH_A, "text": "old"}], keep="yes", collision="old")]
    incoming = [_entry("a.html", 1, HASH_A, chunks=[{"id": HASH_B, "text": "new"}], collision="new", added="yes")]
    old_existing, old_incoming = copy.deepcopy(existing), copy.deepcopy(incoming)
    merged = merge_vault_entries(existing, incoming)
    assert merged[0] is not existing[0]
    assert merged[0]["chunks"] == incoming[0]["chunks"]
    assert merged[0]["keep"] == "yes" and merged[0]["collision"] == "new"
    assert existing == old_existing and incoming == old_incoming
    hashed = [_entry("a.html", 1, HASH_A, chunks=[{"id": HASH_A, "text": "keep"}], keep="yes")]
    missing = [_entry("a.html", 1, None, chunks=[{"id": HASH_B, "text": "lose"}], keep="no")]
    old_hashed, old_missing = copy.deepcopy(hashed), copy.deepcopy(missing)
    kept = merge_vault_entries(hashed, missing)
    assert kept[0] is hashed[0]
    assert kept[0] == old_hashed[0]
    assert hashed == old_hashed and missing == old_missing


def test_schema_backward_compatibility_and_invalid_in_memory_data(tmp_path):
    legacy = [_entry(content_hash=None, extra="ok")]
    atomic_write_json(tmp_path / "legacy.json", legacy)
    assert load_vault(tmp_path / "legacy.json") == legacy
    invalids = [
        {},
        ["entry"],
        [{"file_name": "", "modification_time": 1, "chunks": []}],
        [{"file_name": "a", "modification_time": True, "chunks": []}],
        [{"file_name": "a", "modification_time": float("nan"), "chunks": []}],
        [{"file_name": "a", "modification_time": 1, "chunks": ["chunk"]}],
        [{"file_name": "a", "modification_time": 1, "chunks": [{"id": "x", "text": "x"}]}],
        [{"file_name": "a", "modification_time": 1, "chunks": [], "content_hash": "x"}],
        [{"file_name": "a", "modification_time": 1, "chunks": [], 1: "bad"}],
        [{"file_name": "a", "modification_time": 1, "chunks": [], "unknown": object()}],
    ]
    for invalid in invalids:
        with pytest.raises(ValueError):
            merge_vault_entries(invalid, [])
        with pytest.raises(ValueError):
            atomic_write_json(tmp_path / "missing" / "vault.json", invalid)
    assert not (tmp_path / "missing").exists()


def test_corrupt_input_backup_then_valid_rebuild_is_unique_and_byte_exact(tmp_path):
    vault = tmp_path / "vault.json"
    collision = tmp_path / "vault.json.invalid.20000101T000000000000Z.1.bak"
    collision.write_bytes(b"collision")
    for bad in (b"\xff", b"{bad", b'{"not":"a-list"}'):
        vault.write_bytes(bad)
        assert load_vault(vault) == []
        backups = [p for p in tmp_path.glob("vault.json.invalid.*.bak") if p != collision]
        assert any(p.read_bytes() == bad for p in backups)
    data = _valid_payload()
    atomic_write_json(vault, data)
    assert load_vault(vault) == data


def test_backup_move_failure_leaves_corrupt_target_unchanged(tmp_path, monkeypatch):
    vault = tmp_path / "vault.json"
    old = b"{broken"
    vault.write_bytes(old)
    sentinel = RuntimeError("rename")
    monkeypatch.setattr(vault_store.os, "rename", lambda *_: (_ for _ in ()).throw(sentinel))
    with pytest.raises(RuntimeError) as raised:
        load_vault(vault)
    assert raised.value is sentinel
    assert vault.read_bytes() == old
    assert not list(tmp_path.glob("*.bak"))


def test_dump_encode_write_flush_and_close_failures_preserve_target(tmp_path, monkeypatch):
    for event in ("dump", "write", "flush", "close"):
        vault = tmp_path / f"{event}.json"
        old = b"old target"
        vault.write_bytes(old)
        sentinel = RuntimeError(event)
        original_dump = vault_store.json.dump
        original_temp = vault_store.tempfile.NamedTemporaryFile

        if event == "dump":
            def fail_dump(data, handle, **kwargs):
                handle.write("[")
                raise sentinel
            monkeypatch.setattr(vault_store.json, "dump", fail_dump)
        else:
            class Wrapped:
                def __init__(self, handle): self._handle = handle
                @property
                def name(self): return self._handle.name
                def write(self, value):
                    if event == "write": raise sentinel
                    return self._handle.write(value)
                def flush(self):
                    if event == "flush": raise sentinel
                    return self._handle.flush()
                def fileno(self): return self._handle.fileno()
                def close(self):
                    result = self._handle.close()
                    if event == "close": raise sentinel
                    return result
            def wrapped_temp(*args, **kwargs): return Wrapped(original_temp(*args, **kwargs))
            monkeypatch.setattr(vault_store.tempfile, "NamedTemporaryFile", wrapped_temp)
        with pytest.raises(RuntimeError) as raised:
            atomic_write_json(vault, _valid_payload())
        assert raised.value is sentinel
        assert vault.read_bytes() == old
        assert not list(tmp_path.glob(f".{vault.name}.*.tmp"))
        monkeypatch.setattr(vault_store.json, "dump", original_dump)
        monkeypatch.setattr(vault_store.tempfile, "NamedTemporaryFile", original_temp)


def test_fsync_failure_preserves_target_cleans_temp_and_reraises(tmp_path, monkeypatch):
    vault = tmp_path / "vault.json"; old = b"old"; vault.write_bytes(old); sentinel = RuntimeError("fsync")
    monkeypatch.setattr(vault_store.os, "fsync", lambda *_: (_ for _ in ()).throw(sentinel))
    with pytest.raises(RuntimeError) as raised: atomic_write_json(vault, _valid_payload())
    assert raised.value is sentinel and vault.read_bytes() == old
    assert not list(tmp_path.glob(".vault.json.*.tmp"))


def test_chmod_failure_preserves_target_cleans_temp_and_reraises(tmp_path, monkeypatch):
    vault = tmp_path / "vault.json"; old = b"old"; vault.write_bytes(old); sentinel = RuntimeError("chmod")
    monkeypatch.setattr(vault_store.os, "chmod", lambda *_: (_ for _ in ()).throw(sentinel))
    with pytest.raises(RuntimeError) as raised: atomic_write_json(vault, _valid_payload())
    assert raised.value is sentinel and vault.read_bytes() == old
    assert not list(tmp_path.glob(".vault.json.*.tmp"))


def test_replace_failure_preserves_target_cleans_temp_and_reraises(tmp_path, monkeypatch):
    vault = tmp_path / "vault.json"; old = b"old"; vault.write_bytes(old); sentinel = RuntimeError("replace")
    monkeypatch.setattr(vault_store.os, "replace", lambda *_: (_ for _ in ()).throw(sentinel))
    with pytest.raises(RuntimeError) as raised: atomic_write_json(vault, _valid_payload())
    assert raised.value is sentinel and vault.read_bytes() == old
    assert not list(tmp_path.glob(".vault.json.*.tmp"))


def test_cleanup_failure_preserves_primary_and_records_residual_temp(tmp_path, monkeypatch):
    vault = tmp_path / "vault.json"; old = b"old"; vault.write_bytes(old); sentinel = RuntimeError("replace")
    monkeypatch.setattr(vault_store.os, "replace", lambda *_: (_ for _ in ()).throw(sentinel))
    original_unlink = Path.unlink
    calls = []
    def fail_unlink(path, *args, **kwargs):
        if path.name.startswith(".vault.json."):
            calls.append(path); raise RuntimeError("unlink")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(RuntimeError) as raised: atomic_write_json(vault, _valid_payload())
    assert raised.value is sentinel and vault.read_bytes() == old and len(calls) == 1
    assert any(str(calls[0]) in note and "unlink" in note for note in raised.value.__notes__)
    assert calls[0].exists()


def test_atomic_temp_order_deterministic_bytes_parent_and_mode(tmp_path, monkeypatch):
    events = []
    vault = tmp_path / "nested" / "vault.json"
    temp_parents = []
    original_dump, original_fsync, original_replace, original_chmod = vault_store.json.dump, vault_store.os.fsync, vault_store.os.replace, vault_store.os.chmod
    original_temp = vault_store.tempfile.NamedTemporaryFile
    def recording_dump(*args, **kwargs): events.append("dump"); return original_dump(*args, **kwargs)
    def recording_fsync(*args): events.append("fsync"); return original_fsync(*args)
    def recording_replace(src, dst): events.append("replace"); return original_replace(src, dst)
    def recording_chmod(path, mode): events.append("chmod"); return original_chmod(path, mode)
    class RecordingHandle:
        def __init__(self, handle): self._handle = handle; temp_parents.append(Path(handle.name).parent)
        @property
        def name(self): return self._handle.name
        def write(self, value): return self._handle.write(value)
        def flush(self): events.append("flush"); return self._handle.flush()
        def fileno(self): return self._handle.fileno()
        def close(self): events.append("close"); return self._handle.close()
    monkeypatch.setattr(vault_store.json, "dump", recording_dump); monkeypatch.setattr(vault_store.os, "fsync", recording_fsync); monkeypatch.setattr(vault_store.os, "replace", recording_replace); monkeypatch.setattr(vault_store.os, "chmod", recording_chmod)
    monkeypatch.setattr(vault_store.tempfile, "NamedTemporaryFile", lambda *args, **kwargs: RecordingHandle(original_temp(*args, **kwargs)))
    atomic_write_json(vault, [_entry("é.html")])
    expected = (json.dumps([_entry("é.html")], ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    assert vault.parent.exists() and temp_parents == [vault.parent]
    assert events == ["dump", "flush", "fsync", "close", "replace"]
    assert vault.read_bytes() == expected and b"\r\n" not in expected
    original_mode = stat.S_IMODE(vault.stat().st_mode)
    events.clear(); atomic_write_json(vault, [_entry("é.html")])
    assert temp_parents == [vault.parent, vault.parent]
    assert events == ["dump", "flush", "fsync", "close", "chmod", "replace"]
    assert stat.S_IMODE(vault.stat().st_mode) == original_mode


def test_builder_rerun_and_same_mtime_content_change(tmp_path, monkeypatch):
    import nltk
    monkeypatch.setattr(nltk, "download", lambda *args, **kwargs: None)
    monkeypatch.setattr(nltk.data, "load", lambda *args, **kwargs: type("T", (), {"tokenize": lambda _, text: [text]})())
    sys.modules.pop("Vault_json_creation_from_HTMLs", None)
    import Vault_json_creation_from_HTMLs as builder
    source = tmp_path / "source"; source.mkdir(); html = source / "a.html"; html.write_text("<p>first text.</p>", encoding="utf-8")
    vault = tmp_path / "vault.json"
    monkeypatch.setattr(builder, "_get_sentence_tokenizer", lambda: type("T", (), {"tokenize": lambda _, text: [text]})())
    monkeypatch.setattr(builder, "Pool", lambda *_: type("P", (), {"__enter__": lambda s: s, "__exit__": lambda *a: None, "imap": lambda s, fn, values: map(fn, values)})())
    monkeypatch.setattr(builder, "tqdm", lambda values, **_: values)
    builder.convert_html_to_json(source, vault); first = load_vault(vault)
    builder.convert_html_to_json(source, vault); assert load_vault(vault) == first
    mtime = html.stat().st_mtime; html.write_text("<p>second text.</p>", encoding="utf-8"); os.utime(html, (mtime, mtime))
    builder.convert_html_to_json(source, vault); changed = load_vault(vault)
    assert len(changed) == 1 and changed[0]["content_hash"] != first[0]["content_hash"]


def test_builder_import_is_side_effect_free(tmp_path):
    fake_root = tmp_path / "fake"
    fake = fake_root / "nltk"; fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("from . import data\ndef download(*a, **k): raise AssertionError('download')\n", encoding="utf-8")
    (fake / "data.py").write_text("def load(*a, **k): raise AssertionError('load')\n", encoding="utf-8")
    env = os.environ | {"PYTHONPATH": str(fake_root) + os.pathsep + str(Path.cwd()), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", "import Vault_json_creation_from_HTMLs"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "vault.json").exists()


def test_owned_writer_has_no_append_mode():
    for name in ("vault_store.py", "Vault_json_creation_from_HTMLs.py"):
        tree = ast.parse(Path(name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if callee in {"open", "NamedTemporaryFile"}:
                    modes = [arg.value for arg in node.args[1:2] if isinstance(arg, ast.Constant)]
                    modes += [keyword.value.value for keyword in node.keywords if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant)]
                    assert all("a" not in str(mode) for mode in modes)
    source = Path("vault_store.py").read_text(encoding="utf-8")
    assert source.count("os.replace(") == 1
