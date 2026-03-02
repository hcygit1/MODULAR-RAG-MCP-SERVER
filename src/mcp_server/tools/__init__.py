"""
MCP Tools 模块

定义并注册 MCP Server 暴露的工具。

- execute_* : 内部实现，接收 arguments dict，返回 dict（供单元测试等复用）
- *_hub / list_collections : FastMCP 工具，具名参数，返回 CallToolResult
"""

from src.mcp_server.tools.list_collections import (
    execute_list_collections,
    list_collections,
)
from src.mcp_server.tools.query_knowledge_hub import (
    execute_query_knowledge_hub,
    query_knowledge_hub,
)

__all__ = [
    "execute_list_collections",
    "execute_query_knowledge_hub",
    "list_collections",
    "query_knowledge_hub",
]
