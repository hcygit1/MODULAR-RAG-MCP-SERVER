"""
MCP Tools 统一错误处理

E6：错误处理与协议合规。提供统一的错误返回格式与错误码映射，
确保三个 tool 的错误响应结构一致，便于 Client 程序化解析。
"""
from __future__ import annotations

from typing import Any, Dict, Literal

# 错误类型与 JSON-RPC/MCP 扩展码映射
ErrorType = Literal["INVALID_PARAMS", "RESOURCE_NOT_FOUND", "INTERNAL_ERROR"]

ERROR_CODES: Dict[str, int] = {
    "INVALID_PARAMS": -32602,
    "RESOURCE_NOT_FOUND": -32001,
    "INTERNAL_ERROR": -32603,
}


def build_error_response(
    error_type: ErrorType,
    message: str,
    *,
    structured_content_base: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    构造统一的 MCP tool 错误返回。

    Args:
        error_type: 错误类型，对应 INVALID_PARAMS / RESOURCE_NOT_FOUND / INTERNAL_ERROR
        message: 人类可读的错误说明
        structured_content_base: 错误时需返回的结构化内容骨架（如 citations: []、collections: []），
            会与 errorCode/errorType 合并

    Returns:
        MCP tools/call result 格式：{ content, structuredContent, isError }
    """
    code = ERROR_CODES.get(error_type, -32603)
    sc = dict(structured_content_base) if structured_content_base else {}
    sc["errorCode"] = code
    sc["errorType"] = error_type
    sc["message"] = message
    return {
        "content": [{"type": "text", "text": message}],
        "structuredContent": sc,
        "isError": True,
    }
