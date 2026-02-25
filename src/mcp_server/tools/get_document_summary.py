"""
get_document_summary Tool

按 doc_id 返回 title/summary/tags（从 BM25 chunk metadata 中聚合）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.types import CallToolResult, TextContent

# 默认 BM25 索引路径
_DEFAULT_BM25_BASE_PATH = "data/db/bm25"

_bm25_base_path: str = _DEFAULT_BM25_BASE_PATH


GET_DOCUMENT_SUMMARY_DEFINITION: Dict[str, Any] = {
    "name": "get_document_summary",
    "description": "根据文档 ID 获取文档摘要信息（title、summary、tags）。从已索引的 chunk 元数据中聚合。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "文档唯一标识符",
            },
            "collection_name": {
                "type": "string",
                "description": "集合名称，指定时仅在对应集合中查找（可选）",
            },
        },
        "required": ["doc_id"],
    },
}


def set_bm25_base_path(path: str) -> None:
    """测试注入用：设置 BM25 索引根路径。"""
    global _bm25_base_path
    _bm25_base_path = path


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


def _load_chunk_metadata_for_collection(collection_name: str) -> Dict[str, Dict[str, Any]]:
    """从 BM25 索引文件加载 chunk_metadata。"""
    index_file = Path(_bm25_base_path) / collection_name / "index.json"
    if not index_file.exists():
        return {}
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("chunk_metadata", {})
    except Exception:
        return {}


def _find_doc_metadata(doc_id: str, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    在 chunk_metadata 中查找属于 doc_id 的文档元数据。

    通过 chunk_id 前缀（doc_id_chunk_）或 source_doc_id 匹配。
    返回第一个匹配 chunk 的 metadata（通常含 title/summary/tags/source_path）。
    """
    base = Path(_bm25_base_path)
    if not base.exists():
        return None

    collections: List[str] = []
    if collection_name:
        if (base / collection_name / "index.json").exists():
            collections = [collection_name]
    else:
        collections = [p.name for p in base.iterdir() if p.is_dir() and (p / "index.json").exists()]

    prefix = f"{doc_id}_chunk_"
    for coll in collections:
        chunk_meta = _load_chunk_metadata_for_collection(coll)
        for chunk_id, info in chunk_meta.items():
            meta = info.get("metadata", {}) if isinstance(info, dict) else {}
            if chunk_id.startswith(prefix) or meta.get("source_doc_id") == doc_id:
                return meta
    return None


def _extract_title(meta: Dict[str, Any], doc_id: str) -> str:
    """从 metadata 提取 title，无则从 source_path 或 doc_id 推断。"""
    if meta.get("title"):
        return str(meta["title"])
    source = meta.get("source_path") or meta.get("source_doc_id")
    if source and isinstance(source, str):
        if "/" in source or "\\" in source:
            return source.split("/")[-1].split("\\")[-1]
        return source
    return doc_id


def execute_get_document_summary(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 get_document_summary 工具。

    Args:
        arguments: 包含 doc_id（必填）、collection_name（可选）

    Returns:
        MCP tools/call result：存在时 content + structuredContent；不存在时 isError=True
    """
    doc_id = arguments.get("doc_id")
    if not doc_id or not str(doc_id).strip():
        return {
            "content": [{"type": "text", "text": "参数 doc_id 不能为空"}],
            "structuredContent": {},
            "isError": True,
        }

    doc_id = str(doc_id).strip()
    collection_name = arguments.get("collection_name")
    if collection_name is not None and not isinstance(collection_name, str):
        collection_name = None
    elif collection_name:
        collection_name = collection_name.strip() or None

    try:
        meta = _find_doc_metadata(doc_id, collection_name)
        if meta is None:
            return {
                "content": [{"type": "text", "text": f"文档不存在: {doc_id}"}],
                "structuredContent": {},
                "isError": True,
            }

        title = _extract_title(meta, doc_id)
        summary = meta.get("summary")
        if summary is None:
            summary = ""
        tags = meta.get("tags")
        if not isinstance(tags, list):
            tags = [] if tags is None else [str(tags)]

        result = {
            "doc_id": doc_id,
            "title": title,
            "summary": str(summary) if summary else "",
            "tags": [str(t) for t in tags],
        }
        text = f"标题: {title}\n摘要: {summary or '（无）'}\n标签: {', '.join(tags) or '（无）'}"
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": result,
            "isError": False,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"获取文档摘要失败: {e}"}],
            "structuredContent": {},
            "isError": True,
        }


def get_document_summary(
    doc_id: str,
    collection_name: str | None = None,
) -> CallToolResult:
    """
    根据文档 ID 获取文档摘要信息（title、summary、tags）。从已索引的 chunk 元数据中聚合。

    Args:
        doc_id: 文档唯一标识符
        collection_name: 集合名称，指定时仅在对应集合中查找（可选）

    Returns:
        CallToolResult 含 content、structuredContent（doc_id/title/summary/tags）
    """
    d = execute_get_document_summary({
        "doc_id": doc_id,
        "collection_name": collection_name,
    })
    return _dict_to_call_tool_result(d)
