"""运行目录解析契约。"""

from pathlib import Path

from modular_rag_mcp_server.core.runtime_paths import (
    get_runtime_paths,
    set_data_dir,
)
from modular_rag_mcp_server.core.settings import load_settings


def test_cli_data_dir_has_priority_over_environment(monkeypatch, tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    cli_dir = tmp_path / "cli"
    monkeypatch.setenv("MODULAR_RAG_DATA_DIR", str(env_dir))

    set_data_dir(str(cli_dir))
    try:
        paths = get_runtime_paths()
        assert paths.root == cli_dir
        assert paths.sqlite_path == cli_dir / "data" / "db" / "rag.sqlite"
        assert paths.log_path == cli_dir / "logs" / "traces.jsonl"
        assert paths.integrity_db_path == cli_dir / "cache" / "processing" / "file_integrity.db"
    finally:
        set_data_dir(None)


def test_environment_data_dir_is_used_when_cli_override_is_absent(
    monkeypatch, tmp_path: Path
) -> None:
    env_dir = tmp_path / "env"
    monkeypatch.setenv("MODULAR_RAG_DATA_DIR", str(env_dir))

    set_data_dir(None)
    assert get_runtime_paths().root == env_dir


def test_current_working_directory_remains_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MODULAR_RAG_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    set_data_dir(None)

    assert get_runtime_paths().root == tmp_path


def test_settings_rebases_known_writable_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODULAR_RAG_DATA_DIR", str(tmp_path))
    set_data_dir(None)

    settings = load_settings()

    assert settings.vector_store.sqlite_path == str(tmp_path / "data" / "db" / "rag.sqlite")
    assert settings.observability.logging.log_file == str(tmp_path / "logs" / "traces.jsonl")
