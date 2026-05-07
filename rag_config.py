import os
from pathlib import Path


def get_optional_path(env_name: str) -> Path | None:
    value = os.getenv(env_name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def get_path(env_name: str, default: Path) -> Path:
    return get_optional_path(env_name) or default.expanduser().resolve()
