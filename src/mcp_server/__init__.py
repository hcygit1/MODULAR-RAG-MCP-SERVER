"""
MCP Server 模块

提供 MCP (Model Context Protocol) 协议的 Server 实现，
基于官方 MCP SDK (FastMCP)，支持 Stdio Transport。
"""

from src.mcp_server.server import run_server

__all__ = ["run_server"]
