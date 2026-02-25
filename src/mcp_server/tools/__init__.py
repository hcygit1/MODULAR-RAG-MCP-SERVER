"""
MCP Tools 模块

定义并注册 MCP Server 暴露的工具。
"""

from src.mcp_server.tools.list_collections import (
    LIST_COLLECTIONS_DEFINITION,
    execute_list_collections,
)
from src.mcp_server.tools.query_knowledge_hub import (
    QUERY_KNOWLEDGE_HUB_DEFINITION,
    execute_query_knowledge_hub,
)

__all__ = [
    "LIST_COLLECTIONS_DEFINITION",
    "QUERY_KNOWLEDGE_HUB_DEFINITION",
    "execute_list_collections",
    "execute_query_knowledge_hub",
]
