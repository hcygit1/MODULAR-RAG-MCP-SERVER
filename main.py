#!/usr/bin/env python3
"""
MCP Server 启动入口

遵循 MCP 协议规范，通过 Stdio Transport 与 MCP Clients 通信。
"""
import sys
import logging
from pathlib import Path

# 配置日志输出到 stderr（避免污染 stdout 的 MCP 消息）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


def main():
    """MCP Server 主入口"""
    logger.info("Starting Modular RAG MCP Server...")
    
    # TODO: 在 A3 阶段实现配置加载
    # from src.core.settings import load_settings
    # settings = load_settings("config/settings.yaml")
    
    # TODO: 在 E1 阶段实现 MCP Server
    # from src.mcp_server.server import create_server
    # server = create_server(settings)
    # server.run()
    
    logger.info("Server startup placeholder - implementation pending")
    print('{"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not implemented"}}', file=sys.stderr)


if __name__ == "__main__":
    main()
