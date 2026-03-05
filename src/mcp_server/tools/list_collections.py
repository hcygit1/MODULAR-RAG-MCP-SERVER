"""
list_collections Tool

列出集合（与 query、ingest 数据源一致）。
统一存储：从 SQLite chunks 表或 VectorStore.list_collections 读取。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.types import CallToolResult

from src.mcp_server.tools.config_utils import load_mcp_settings
from src.mcp_server.tools.mcp_utils import dict_to_call_tool_result

# 测试注入：非 None 时直接使用 sqlite_path，否则从 settings 读取
_inject_sqlite_path: Optional[str] = None


def set_base_path(path: Optional[str]) -> None:
    """测试注入用：设置 SQLite 路径。传入 None 可恢复为从 settings 读取。"""
    global _inject_sqlite_path
    _inject_sqlite_path = path


def _list_collections_from_sqlite(sqlite_path: str) -> List[str]:
    """从 SQLite chunks 表列出集合。"""
    try:
        conn = sqlite3.connect(str(sqlite_path))
        rows = conn.execute(
            "SELECT DISTINCT collection_name FROM chunks ORDER BY collection_name"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def execute_list_collections(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 list_collections 工具。

    Args:
        arguments: 可选 base_path（已废弃，保留兼容），不传则从 settings.vector_store.sqlite_path 读取

    Returns:
        MCP tools/call result：{ content, structuredContent: { collections }, isError }
    """
    try:
        settings = load_mcp_settings()
        sqlite_path = _inject_sqlite_path or getattr(
            settings.vector_store, "sqlite_path", None
        )
        if sqlite_path:
            collections = _list_collections_from_sqlite(sqlite_path)
        else:
            collections = []

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


def list_collections(base_path: Optional[str] = None) -> CallToolResult:
    """
    列出知识库中的集合名称。数据源自 SQLite chunks 表（与 query、ingest 一致）。

    Args:
        base_path: 已废弃，保留兼容。不传则从 config 的 vector_store.sqlite_path 读取

    Returns:
        CallToolResult 含 content（文本）、structuredContent.collections
    """
    args: Dict[str, Any] = {}
    d = execute_list_collections(args)
    return dict_to_call_tool_result(d)
