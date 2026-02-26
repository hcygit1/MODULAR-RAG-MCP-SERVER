"""
MCP Tools 公共工具

提供 dict → CallToolResult 的统一转换，供 query_knowledge_hub、list_collections、
get_document_summary 复用，消除重复的 _dict_to_call_tool_result 实现。
"""
from __future__ import annotations

from typing import Any, Dict

from mcp.types import CallToolResult, ImageContent, TextContent


def dict_to_call_tool_result(d: Dict[str, Any]) -> CallToolResult:
    """
    将内部 dict 格式转为 MCP SDK 的 CallToolResult。

    支持 TextContent 与 ImageContent。内部 dict 格式：
    { content: [{type, text} | {type, data, mimeType}], structuredContent, isError }

    Args:
        d: 工具执行返回的 dict，含 content、structuredContent、isError

    Returns:
        CallToolResult 供 FastMCP 序列化返回给 Client
    """
    content_blocks = []
    for c in d.get("content", []):
        ctype = c.get("type", "text")
        if ctype == "image":
            content_blocks.append(
                ImageContent(
                    type="image",
                    data=c.get("data", ""),
                    mimeType=c.get("mimeType", "image/png"),
                )
            )
        else:
            content_blocks.append(TextContent(type="text", text=c.get("text", "")))
    return CallToolResult(
        content=content_blocks,
        structuredContent=d.get("structuredContent") or {},
        isError=d.get("isError", False),
    )
