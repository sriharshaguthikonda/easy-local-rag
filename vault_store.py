import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def load_vault(path: str | Path) -> list[dict[str, Any]]:
    vault_path = Path(path)
    if not vault_path.exists() or vault_path.stat().st_size == 0:
        return []

    try:
        with vault_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        backup_name = (
            f"{vault_path.name}.invalid.{int(time.time())}.bak"
        )
        backup_path = vault_path.with_name(backup_name)
        os.replace(vault_path, backup_path)
        print(f"Invalid vault JSON moved to backup: {backup_path}")
        return []

    if isinstance(data, list):
        return data

    print(f"Vault JSON is not a list in {vault_path}; starting with empty list.")
    return []


def merge_vault_entries(
    existing_entries: list[dict[str, Any]],
    new_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for entry in existing_entries:
        file_name = entry.get("file_name")
        if file_name:
            merged[os.path.normpath(file_name)] = entry

    for entry in new_entries:
        file_name = entry.get("file_name")
        if not file_name:
            continue
        norm_file = os.path.normpath(file_name)
        existing = merged.get(norm_file)
        if not existing or existing.get("modification_time") != entry.get(
            "modification_time"
        ):
            merged[norm_file] = entry

    return list(merged.values())


def atomic_write_json(path: str | Path, data: Any) -> None:
    vault_path = Path(path)
    vault_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=vault_path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        temp_path = Path(handle.name)

    os.replace(temp_path, vault_path)
