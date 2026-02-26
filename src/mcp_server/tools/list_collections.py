"""
list_collections Tool

列出 data/documents/ 下集合并附带统计（统计可延后）。
每个子目录视为一个集合。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from mcp.types import CallToolResult, TextContent

# 默认文档根路径
_DEFAULT_DOCUMENTS_PATH = "data/documents"

# 测试注入用
_base_path: str = _DEFAULT_DOCUMENTS_PATH


# 工具 schema
LIST_COLLECTIONS_DEFINITION: Dict[str, Any] = {
    "name": "list_collections",
    "description": "列出知识库中的集合名称。每个集合对应 data/documents/ 下的一个子目录，或已构建的索引集合。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "base_path": {
                "type": "string",
                "description": "文档根路径，默认为 data/documents",
            },
        },
        "required": [],
    },
}


def set_base_path(path: str) -> None:
    """测试注入用：设置文档根路径。"""
    global _base_path
    _base_path = path


def _dict_to_call_tool_result(d: Dict[str, Any]) -> CallToolResult:
    """将 dict 格式转为 CallToolResult。"""
    content = [
        TextContent(type=c.get("type", "text"), text=c.get("text", ""))
        for c in d.get("content", [])
    ]
    return CallToolResult(
        content=content,
        structuredContent=d.get("structuredContent") or {},
        isError=d.get("isError", False),
    )


def _list_collections_from_fs(base_path: str) -> List[str]:
    """
    从文件系统列出集合（子目录名）。

    Args:
        base_path: 根路径，如 data/documents

    Returns:
        集合名列表，按字母序
    """
    root = Path(base_path)
    if not root.exists() or not root.is_dir():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(names)


def execute_list_collections(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 list_collections 工具。

    Args:
        arguments: 可选 base_path

    Returns:
        MCP tools/call result：{ content, structuredContent: { collections }, isError }
    """
    base = arguments.get("base_path") if isinstance(arguments.get("base_path"), str) else _base_path
    base = base.strip() if base else _base_path
    if not base:
        base = _DEFAULT_DOCUMENTS_PATH

    try:
        collections = _list_collections_from_fs(base)
        text = ", ".join(collections) if collections else "（暂无集合）"
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": {"collections": collections},
            "isError": False,
        }
    except Exception as e:
        from src.mcp_server.tools.error_utils import build_error_response

        return build_error_response(
            "INTERNAL_ERROR",
            f"列出集合失败: {e}",
            structured_content_base={"collections": []},
        )


def list_collections(base_path: str = "data/documents") -> CallToolResult:
    """
    列出知识库中的集合名称。每个集合对应 data/documents/ 下的一个子目录，或已构建的索引集合。

    Args:
        base_path: 文档根路径，默认为 data/documents

    Returns:
        CallToolResult 含 content（文本）、structuredContent.collections
    """
    d = execute_list_collections({"base_path": base_path})
    return _dict_to_call_tool_result(d)
