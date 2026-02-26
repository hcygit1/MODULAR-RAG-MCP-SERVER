#!/usr/bin/env python3
"""
启动 Trace Dashboard

读取 config/settings.yaml 中的端口配置，调用 streamlit 启动 Dashboard。

用法:
    python scripts/start_dashboard.py
    python scripts/start_dashboard.py --port 8502
    python scripts/start_dashboard.py --log-file ./logs/traces.jsonl
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
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
    args = parser.parse_args()

    port = args.port
    if port is None:
        try:
            from src.core.settings import load_settings
            settings = load_settings("config/settings.yaml")
            port = settings.observability.dashboard.port
        except Exception:
            port = 8501

    app_path = Path(__file__).resolve().parent.parent / "src" / "observability" / "dashboard" / "app.py"
    if not app_path.exists():
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
