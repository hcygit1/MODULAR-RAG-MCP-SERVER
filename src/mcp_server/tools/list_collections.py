"""
list_collections Tool

列出 BM25 索引目录下的集合（与 query、ingest 数据源一致）。
每个子目录视为一个集合，数据源自 settings.ingestion.bm25_base_path。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.types import CallToolResult

from src.mcp_server.tools.config_utils import load_mcp_settings
from src.mcp_server.tools.mcp_utils import dict_to_call_tool_result

# 测试注入用：非 None 时直接使用，否则从 settings 读取
_base_path: Optional[str] = None


def set_base_path(path: Optional[str]) -> None:
    """测试注入用：设置文档根路径。传入 None 可恢复为从 settings 读取。"""
    global _base_path
    _base_path = path


def _get_default_base_path() -> str:
    """从 settings 读取 bm25_base_path，与 query_knowledge_hub、ingest 一致。"""
    settings = load_mcp_settings()
    return settings.ingestion.bm25_base_path


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
        arguments: 可选 base_path，不传则从 settings.ingestion.bm25_base_path 读取

    Returns:
        MCP tools/call result：{ content, structuredContent: { collections }, isError }
    """
    arg_base = arguments.get("base_path")
    if isinstance(arg_base, str) and arg_base.strip():
        base = arg_base.strip()
    elif _base_path is not None:
        base = _base_path
    else:
        base = _get_default_base_path()

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


def list_collections(base_path: Optional[str] = None) -> CallToolResult:
    """
    列出知识库中的集合名称。数据源自 BM25 索引目录（与 query、ingest 一致）。

    Args:
        base_path: 索引根路径，不传则从 config 的 bm25_base_path 读取

    Returns:
        CallToolResult 含 content（文本）、structuredContent.collections
    """
    args: Dict[str, Any] = (
        {} if base_path is None or (isinstance(base_path, str) and not base_path.strip()) else {"base_path": base_path.strip()}
    )
    d = execute_list_collections(args)
    return dict_to_call_tool_result(d)
