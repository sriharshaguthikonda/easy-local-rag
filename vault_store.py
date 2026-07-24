import json
import math
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _validate_vault_data(data: object) -> None:
    def validate_value(value: object) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("vault data contains a non-finite number")
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("vault object keys must be strings")
            for nested in value.values():
                validate_value(nested)
        elif isinstance(value, list):
            for nested in value:
                validate_value(nested)

    if not isinstance(data, list):
        raise ValueError("vault data must be a list")
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("vault entries must be objects")
        file_name = entry.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("vault entry file_name must be a non-empty string")
        modification_time = entry.get("modification_time")
        if isinstance(modification_time, bool) or not isinstance(modification_time, (int, float)) or not math.isfinite(modification_time):
            raise ValueError("vault entry modification_time must be finite")
        chunks = entry.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("vault entry chunks must be a list")
        content_hash = entry.get("content_hash")
        if content_hash is not None and (not isinstance(content_hash, str) or len(content_hash) != 64 or any(character not in "0123456789abcdef" for character in content_hash)):
            raise ValueError("vault entry content_hash must be a lowercase SHA-256")
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("vault chunks must be objects")
            chunk_id, text = chunk.get("id"), chunk.get("text")
            if not isinstance(chunk_id, str) or len(chunk_id) != 64 or any(character not in "0123456789abcdef" for character in chunk_id) or not isinstance(text, str):
                raise ValueError("vault chunks require a lowercase SHA-256 id and text")
        validate_value(entry)
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("vault data is not JSON serializable") from error


def _normalized_file_identity(file_name: str) -> str:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("vault file_name must be a non-empty string")
    return os.path.normcase(os.path.abspath(os.path.normpath(file_name)))


def _invalid_backup_path(vault_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{vault_path.name}.invalid.{timestamp}.{os.getpid()}"
    candidate = vault_path.with_name(f"{stem}.bak")
    counter = 1
    while candidate.exists():
        candidate = vault_path.with_name(f"{stem}.{counter}.bak")
        counter += 1
    return candidate


def load_vault(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    vault_path = Path(path)
    if not vault_path.exists() or vault_path.stat().st_size == 0:
        return []
    try:
        with vault_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        _validate_vault_data(data)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        backup_path = _invalid_backup_path(vault_path)
        os.rename(vault_path, backup_path)
        print(f"Invalid vault JSON moved to backup: {backup_path}")
        return []
    return data


def merge_vault_entries(
    existing_entries: list[dict[str, Any]],
    new_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _validate_vault_data(existing_entries)
    _validate_vault_data(new_entries)
    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for entry in existing_entries:
        identity = _normalized_file_identity(entry["file_name"])
        if identity in positions:
            merged[positions[identity]] = entry
        else:
            positions[identity] = len(merged)
            merged.append(entry)
    incoming: list[tuple[str, dict[str, Any]]] = []
    incoming_positions: dict[str, int] = {}
    for entry in new_entries:
        identity = _normalized_file_identity(entry["file_name"])
        if identity in incoming_positions:
            incoming[incoming_positions[identity]] = (identity, entry)
        else:
            incoming_positions[identity] = len(incoming)
            incoming.append((identity, entry))
    for identity, entry in incoming:
        position = positions.get(identity)
        if position is None:
            positions[identity] = len(merged)
            merged.append(entry)
            continue
        existing = merged[position]
        old_hash = existing.get("content_hash")
        new_hash = entry.get("content_hash")
        if existing["modification_time"] == entry["modification_time"] and (
            old_hash == new_hash or (old_hash is not None and new_hash is None)
        ):
            continue
        replacement = {key: value for key, value in existing.items() if key not in {"file_name", "modification_time", "chunks", "content_hash"}}
        replacement.update(entry)
        merged[position] = replacement
    return merged


def atomic_write_json(path: str | os.PathLike[str], data: list[dict[str, Any]]) -> None:
    _validate_vault_data(data)
    vault_path = Path(path)
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    saved_mode = stat.S_IMODE(vault_path.stat().st_mode) if vault_path.exists() else None
    handle = None
    temp_path: Path | None = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=f".{vault_path.name}.",
            suffix=".tmp", dir=vault_path.parent, delete=False,
        )
        temp_path = Path(handle.name)
        json.dump(data, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        if saved_mode is not None:
            os.chmod(temp_path, saved_mode)
        os.replace(temp_path, vault_path)
        temp_path = None
    except BaseException as error:
        notes: list[str] = []
        if handle is not None:
            try:
                handle.close()
            except BaseException as cleanup_error:
                notes.append(f"temp close cleanup failed: {cleanup_error!r}")
        if temp_path is not None:
            try:
                temp_path.unlink()
            except BaseException as cleanup_error:
                notes.append(f"residual temp path {temp_path}: cleanup failed: {cleanup_error!r}")
        for note in notes:
            error.add_note(note)
        raise
