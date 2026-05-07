from pathlib import Path

from rag_config import get_optional_path, get_path


def test_get_optional_path_returns_none_for_missing_env(monkeypatch):
    monkeypatch.delenv("EASY_RAG_TEST_PATH", raising=False)
    assert get_optional_path("EASY_RAG_TEST_PATH") is None


def test_get_optional_path_resolves_value(monkeypatch, tmp_path):
    target = tmp_path / "data"
    monkeypatch.setenv("EASY_RAG_TEST_PATH", str(target))
    resolved = get_optional_path("EASY_RAG_TEST_PATH")
    assert resolved == target.resolve()


def test_get_path_returns_default_when_env_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("EASY_RAG_DEFAULT_PATH", raising=False)
    default = tmp_path / "fallback"
    assert get_path("EASY_RAG_DEFAULT_PATH", default) == default.resolve()
