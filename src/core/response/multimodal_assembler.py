"""
多模态内容组装

当检索结果 chunk 含 image_refs 时，从 SQLite images 表读取图片并 base64 编码，
供 MCP tools/call 的 content 中返回 ImageContent。
Phase C：仅支持 sqlite_path，不再支持文件系统 images_base_path。
"""
import base64
import logging
import sqlite3
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import QueryResult

logger = logging.getLogger(__name__)


def _load_image_from_sqlite(
    sqlite_path: str,
    image_id: str,
    collection_name: str,
) -> Optional[tuple]:
    """
    从 images 表加载单张图片。
    Returns:
        (image_data: bytes, mime_type: str) 或 None
    """
    try:
        conn = sqlite3.connect(str(sqlite_path))
        row = conn.execute(
            "SELECT image_data, mime_type FROM images WHERE id = ? AND collection_name = ?",
            (image_id, collection_name),
        ).fetchone()
        conn.close()
        if row:
            return (row[0], row[1] or "image/png")
    except Exception as e:
        logger.warning("从 SQLite 加载图片失败 %s: %s", image_id, e)
    return None


def _infer_collection_from_sqlite(sqlite_path: str, image_id: str) -> Optional[str]:
    """从 images 表推断 image_id 所属的 collection_name。"""
    try:
        conn = sqlite3.connect(str(sqlite_path))
        row = conn.execute(
            "SELECT collection_name FROM images WHERE id = ? LIMIT 1",
            (image_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _image_refs_to_content_items(
    image_refs: List[str],
    collection_name: str,
    sqlite_path: str,
) -> List[Dict[str, Any]]:
    """
    将 image_refs 转为 ImageContent 的 dict 列表。仅从 SQLite images 表加载。

    Args:
        image_refs: 图片 ID 列表
        collection_name: 集合名称
        sqlite_path: SQLite 路径，从 images 表加载

    Returns:
        [{"type": "image", "data": base64_str, "mimeType": "image/png"}, ...]
    """
    items: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for image_id in image_refs:
        if not image_id or image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        loaded = _load_image_from_sqlite(sqlite_path, image_id, collection_name)
        if loaded:
            raw, mime_type = loaded
            b64 = base64.b64encode(raw).decode("ascii")
            items.append({"type": "image", "data": b64, "mimeType": mime_type or "image/png"})
    return items


def assemble_content(
    results: List["QueryResult"],
    text_content: str,
    collection_name: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    组装 MCP content 列表：TextContent + ImageContent（当 metadata 含 image_refs 时）。

    Phase C：仅支持从 sqlite_path 的 images 表加载图片，不再支持文件系统。

    Args:
        results: 检索结果
        text_content: 已有的 Markdown 文本（作为第一个 content 项）
        collection_name: 集合名称；若 None 且 sqlite_path 非空则按 image_id 从 images 表推断
        sqlite_path: SQLite 路径，非空时从 images 表加载图片

    Returns:
        content 列表，如 [{"type":"text","text":...}, {"type":"image","data":...,"mimeType":...}]
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_content}]
    if not sqlite_path:
        return content

    seen_ids: set = set()

    for r in results:
        meta = r.metadata or {}
        refs = meta.get("image_refs")
        if not isinstance(refs, list):
            continue
        coll = collection_name or meta.get("collection_name")
        for image_id in refs:
            if not image_id or image_id in seen_ids:
                continue
            seen_ids.add(image_id)
            if not coll:
                coll = _infer_collection_from_sqlite(sqlite_path, image_id)
            if not coll:
                continue
            items = _image_refs_to_content_items([image_id], coll, sqlite_path)
            content.extend(items)
    return content
