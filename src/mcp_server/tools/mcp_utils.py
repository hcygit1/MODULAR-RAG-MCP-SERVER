"""
MCP Tools 公共工具

提供 dict → CallToolResult 的统一转换，供 query_knowledge_hub、list_collections 复用。
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from typing import Any, Callable, Dict, Optional, TypeVar

from mcp.types import CallToolResult, ImageContent, TextContent

T = TypeVar("T")


async def run_sync_callable(
    func: Callable[..., T],
    *args: Any,
    executor: Optional[Executor] = None,
) -> T:
    """
    在线程 executor 中执行同步操作，并将取消延迟到同步操作结束。

    Python 线程无法安全强制取消。若调用协程被取消，继续等待同步操作到达
    完成或失败的安全边界，再向调用方传播 CancelledError，避免调用方在后台
    写入仍进行时立即重试。
    """
    loop = asyncio.get_running_loop()
    operation = loop.run_in_executor(executor, func, *args)
    cancelled: Optional[asyncio.CancelledError] = None

    while True:
        try:
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError as exc:
            if operation.done():
                if operation.cancelled():
                    if cancelled is not None:
                        raise cancelled from exc
                    raise
                operation_exception = operation.exception()
                if operation_exception is exc:
                    if cancelled is not None:
                        raise cancelled from exc
                    raise
            if cancelled is None:
                cancelled = exc
        except BaseException as exc:
            if cancelled is not None:
                raise cancelled from exc
            raise

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
