from pathlib import Path, PurePosixPath
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".serena",
    "chroma",
    "chroma_db",
    "Embeddings",
    "volumes",
    "your_db_path",
}
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".db",
    ".duckdb",
    ".log",
    ".npy",
    ".npz",
    ".pickle",
    ".pt",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".sst",
    ".wal",
}
FORBIDDEN_NAMES = {"semantic_vault.json", "vault.json", "vault.txt"}


def test_generated_runtime_artifacts_are_not_tracked():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    tracked = [
        PurePosixPath(path) for path in result.stdout.decode().split("\0") if path
    ]
    forbidden = [
        str(path)
        for path in tracked
        if path.parts[:2] != ("tests", "fixtures")
        and (
            FORBIDDEN_DIRS.intersection(path.parts)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or path.name in FORBIDDEN_NAMES
            or (len(path.parts) == 1 and path.name.startswith("ffcache"))
        )
    ]
    assert forbidden == []
