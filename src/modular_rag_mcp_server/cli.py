"""Command-line entry point for the MCP server."""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Any

from modular_rag_mcp_server.mcp_server.tools.config_utils import (
    load_mcp_settings,
    set_config_path,
)

logger = logging.getLogger(__name__)

_COMMAND_NAMES = {
    "ingest",
    "ingest-mineru",
    "retrieve",
    "evaluate",
    "dashboard",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modular-rag-mcp-server",
        description="Run the Modular RAG MCP server over stdio.",
        epilog=(
            "Commands: ingest, ingest-mineru, retrieve, evaluate, dashboard. "
            "Run a command with --help for its options."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Use an external settings YAML instead of the bundled defaults.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and exit without starting the server.",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def _init_observability(settings: Any) -> None:
    observability = getattr(settings, "observability", None)
    logging_config = getattr(observability, "logging", None)
    if logging_config is None:
        return

    from modular_rag_mcp_server.observability.logger import init_trace_collector

    init_trace_collector(
        getattr(logging_config, "log_file", "./logs/traces.jsonl")
    )


def run_server() -> None:
    """Import the MCP runtime only when the server is actually started."""
    from modular_rag_mcp_server.mcp_server.server import run_server as _run_server

    _run_server()


def _run_command(argv: Sequence[str]) -> int | None:
    """Dispatch packaged operational commands without importing them at startup."""
    if not argv or argv[0] not in _COMMAND_NAMES:
        return None

    command = argv[0]
    if command == "ingest":
        from modular_rag_mcp_server.cli_commands.ingest import main as command_main
    elif command == "ingest-mineru":
        from modular_rag_mcp_server.cli_commands.ingest_mineru import main as command_main
    elif command == "retrieve":
        from modular_rag_mcp_server.cli_commands.retrieve import main as command_main
    elif command == "evaluate":
        from modular_rag_mcp_server.cli_commands.evaluate import main as command_main
    else:
        from modular_rag_mcp_server.cli_commands.dashboard import main as command_main

    try:
        command_main(argv[1:])
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration and optionally start the stdio MCP server."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    command_exit_code = _run_command(raw_argv)
    if command_exit_code is not None:
        return command_exit_code

    args = _build_parser().parse_args(raw_argv)
    _configure_logging()
    if args.config:
        set_config_path(args.config)

    try:
        settings = load_mcp_settings()
        logger.info(
            "配置加载成功: LLM provider=%s, Embedding provider=%s",
            settings.llm.provider,
            settings.embedding.provider,
        )
        if args.check_config:
            return 0

        _init_observability(settings)
        logger.info("MCP Server 启动中 (stdio transport)...")
        run_server()
        return 0
    except FileNotFoundError as exc:
        logger.error("配置文件未找到: %s", exc)
    except ValueError as exc:
        logger.error("配置校验失败: %s", exc)
    except Exception as exc:
        logger.error("启动失败: %s", exc, exc_info=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
