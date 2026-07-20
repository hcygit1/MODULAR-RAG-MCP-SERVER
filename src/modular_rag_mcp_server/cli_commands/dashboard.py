#!/usr/bin/env python3
"""
启动 Trace Dashboard

读取默认或外部配置中的端口设置，调用 streamlit 启动 Dashboard。

用法:
    modular-rag-mcp-server dashboard
    modular-rag-mcp-server dashboard --port 8502
    modular-rag-mcp-server dashboard --log-file ./logs/traces.jsonl
"""
import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def get_dashboard_app_path() -> Path:
    """Resolve the dashboard module from the active installed package."""
    spec = importlib.util.find_spec(
        "modular_rag_mcp_server.observability.dashboard.app"
    )
    if spec is None or spec.origin is None:
        raise FileNotFoundError("Dashboard 应用模块不可用")
    return Path(spec.origin)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="启动 RAG Trace Dashboard")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard 端口（默认从 settings.yaml 读取，或 8501）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Trace 日志文件路径（默认 ./logs/traces.jsonl）",
    )
    args = parser.parse_args(argv)

    port = args.port
    if port is None:
        try:
            from modular_rag_mcp_server.mcp_server.tools.config_utils import load_mcp_settings
            settings = load_mcp_settings()
            port = settings.observability.dashboard.port
        except Exception as e:
            logger.warning("读取 Dashboard 端口配置失败，使用默认 8501: %s", e)
            port = 8501

    try:
        app_path = get_dashboard_app_path()
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
    if not app_path.is_file():
        print(f"错误: Dashboard 应用未找到: {app_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
    ]

    if args.log_file:
        cmd.extend(["--", f"--log-file={args.log_file}"])

    print(f"启动 Dashboard: http://localhost:{port}")
    print(f"Trace 日志: {args.log_file or './logs/traces.jsonl'}")
    print("按 Ctrl+C 停止\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard 已停止")
    except FileNotFoundError:
        print("错误: streamlit 未安装，请运行: pip install streamlit", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
