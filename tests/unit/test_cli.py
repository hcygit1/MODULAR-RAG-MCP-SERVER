"""Formal console entry point contracts."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(provider="fake"),
        embedding=SimpleNamespace(provider="fake"),
        observability=None,
    )


def test_check_config_uses_explicit_path_without_starting_server(
    monkeypatch,
) -> None:
    cli = importlib.import_module("modular_rag_mcp_server.cli")
    set_config_path = MagicMock()
    load_settings = MagicMock(return_value=_settings())
    run_server = MagicMock()
    monkeypatch.setattr(cli, "set_config_path", set_config_path)
    monkeypatch.setattr(cli, "load_mcp_settings", load_settings)
    monkeypatch.setattr(cli, "run_server", run_server)

    exit_code = cli.main(["--config", "/tmp/settings.yaml", "--check-config"])

    assert exit_code == 0
    set_config_path.assert_called_once_with("/tmp/settings.yaml")
    load_settings.assert_called_once_with()
    run_server.assert_not_called()


def test_cli_starts_server_after_loading_settings(monkeypatch) -> None:
    cli = importlib.import_module("modular_rag_mcp_server.cli")
    run_server = MagicMock()
    monkeypatch.setattr(cli, "load_mcp_settings", lambda: _settings())
    monkeypatch.setattr(cli, "run_server", run_server)

    exit_code = cli.main([])

    assert exit_code == 0
    run_server.assert_called_once_with()


def test_cli_returns_nonzero_when_configuration_is_invalid(monkeypatch) -> None:
    cli = importlib.import_module("modular_rag_mcp_server.cli")
    run_server = MagicMock()
    monkeypatch.setattr(
        cli,
        "load_mcp_settings",
        MagicMock(side_effect=ValueError("invalid settings")),
    )
    monkeypatch.setattr(cli, "run_server", run_server)

    exit_code = cli.main(["--check-config"])

    assert exit_code == 1
    run_server.assert_not_called()


def test_cli_reports_malformed_yaml_as_configuration_error(
    caplog,
    tmp_path: Path,
) -> None:
    cli = importlib.import_module("modular_rag_mcp_server.cli")
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("llm: [\n", encoding="utf-8")

    try:
        with caplog.at_level(logging.ERROR):
            exit_code = cli.main(["--config", str(config_path), "--check-config"])
    finally:
        cli.set_config_path(None)

    assert exit_code == 1
    assert "配置校验失败" in caplog.text
    assert "启动失败" not in caplog.text


def test_cli_dispatches_packaged_ingest_command(monkeypatch) -> None:
    cli = importlib.import_module("modular_rag_mcp_server.cli")
    command = importlib.import_module("modular_rag_mcp_server.cli_commands.ingest")
    command_main = MagicMock()
    monkeypatch.setattr(command, "main", command_main)

    exit_code = cli.main(["ingest", "--collection", "docs", "--path", "sample.pdf"])

    assert exit_code == 0
    command_main.assert_called_once_with(
        ["--collection", "docs", "--path", "sample.pdf"]
    )


def test_cli_extracts_data_dir_before_subcommand(monkeypatch) -> None:
    cli = importlib.import_module("modular_rag_mcp_server.cli")
    command = importlib.import_module("modular_rag_mcp_server.cli_commands.ingest")
    command_main = MagicMock()
    set_data_dir = MagicMock()
    monkeypatch.setattr(command, "main", command_main)
    monkeypatch.setattr(cli, "set_data_dir", set_data_dir)

    assert cli.main(["--data-dir", "/tmp/rag", "ingest", "--help"]) == 0
    set_data_dir.assert_called_once_with("/tmp/rag")
    command_main.assert_called_once_with(["--help"])
