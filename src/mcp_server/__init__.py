"""
MCP Server 模块

提供 MCP (Model Context Protocol) 协议的 Server 实现，
支持 Stdio Transport：stdin 读取请求，stdout 输出响应，stderr 输出日志。
"""

from src.mcp_server.server import run_server

__all__ = ["run_server"]
