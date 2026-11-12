"""
MCP Tools 公共工具

提供 dict → CallToolResult 的统一转换，供 query_knowledge_hub、list_collections 复用。
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar

from mcp.types import CallToolResult, ImageContent, TextContent

if TYPE_CHECKING:
    from mcp.types import ContentBlock

T = TypeVar("T")


async def run_sync_callable(
    func: Callable[..., T],
    *args: Any,
    executor: Optional[Executor] = None,
) -> T:
    """
    在线程 executor 中执行同步操作，并将取消延迟到同步操作结束。

    Python 线程无法安全强制取消。若调用协程被取消，继续等待同步操作到达
    完成或失败的安全边界，避免调用方在后台写入仍进行时立即重试。操作
    成功后传播 CancelledError；操作失败时优先传播原始异常，防止 Python
    3.10 的 Task 取消语义丢失异常因果链。
    """
    loop = asyncio.get_running_loop()
    operation = loop.run_in_executor(executor, func, *args)
    cancelled: Optional[asyncio.CancelledError] = None

    while True:
        try:
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError as exc:
            if cancelled is None:
                cancelled = exc
            if operation.done():
                if operation.cancelled():
                    raise cancelled from exc
                operation_exception = operation.exception()
                if operation_exception is not None:
                    raise operation_exception
                break

    if cancelled is not None:
        raise cancelled
    return result


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
    content_blocks: list[ContentBlock] = []
    for c in d.get("content", []):
        ctype = c.get("type", "text")
        if ctype == "image":
            content_blocks.append(
                ImageContent(
                    type="image",
                    data=c.get("data", ""),
                    mime_type=c.get("mimeType", "image/png"),
                )
            )
        else:
            content_blocks.append(TextContent(type="text", text=c.get("text", "")))
    return CallToolResult(
        content=content_blocks,
        structured_content=d.get("structuredContent") or {},
        is_error=d.get("isError", False),
    )
