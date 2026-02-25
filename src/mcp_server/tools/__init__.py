"""
MCP Tools 模块

定义并注册 MCP Server 暴露的工具。

- execute_* : 内部实现，接收 arguments dict，返回 dict（供单元测试等复用）
- *_hub / list_collections / get_document_summary : FastMCP 工具，具名参数，返回 CallToolResult
"""

from src.mcp_server.tools.get_document_summary import (
    GET_DOCUMENT_SUMMARY_DEFINITION,
    execute_get_document_summary,
    get_document_summary,
)
from src.mcp_server.tools.list_collections import (
    LIST_COLLECTIONS_DEFINITION,
    execute_list_collections,
    list_collections,
)
from src.mcp_server.tools.query_knowledge_hub import (
    QUERY_KNOWLEDGE_HUB_DEFINITION,
    execute_query_knowledge_hub,
    query_knowledge_hub,
)

__all__ = [
    "GET_DOCUMENT_SUMMARY_DEFINITION",
    "LIST_COLLECTIONS_DEFINITION",
    "QUERY_KNOWLEDGE_HUB_DEFINITION",
    "execute_get_document_summary",
    "execute_list_collections",
    "execute_query_knowledge_hub",
    "get_document_summary",
    "list_collections",
    "query_knowledge_hub",
]
