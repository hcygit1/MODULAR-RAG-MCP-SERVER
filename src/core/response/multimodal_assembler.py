"""
多模态内容组装

当检索结果 chunk 含 image_refs 时，读取图片并 base64 编码，
供 MCP tools/call 的 content 中返回 ImageContent。
"""
import base64
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import QueryResult

logger = logging.getLogger(__name__)


def _load_image_index(images_base_path: str, collection_name: str) -> Dict[str, Dict[str, Any]]:
    """
    从 data/images/{collection}/index.json 加载图片索引。

    Returns:
        images 字典：{ image_id: { file_path, mime_type, ... } }
    """
    base = Path(images_base_path)
    index_file = base / collection_name / "index.json"
    if not index_file.exists():
        return {}
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images", {})
        return images if isinstance(images, dict) else {}
    except Exception as e:
        logger.warning("加载图片索引失败 %s: %s", index_file, e)
        return {}


def _image_refs_to_content_items(
    image_refs: List[str],
    collection_name: str,
    images_base_path: str,
    image_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    将 image_refs 转为 ImageContent 的 dict 列表。

    Args:
        image_refs: 图片 ID 列表
        collection_name: 集合名称
        images_base_path: 图片根路径
        image_index: 已加载的 images 索引

    Returns:
        [{"type": "image", "data": base64_str, "mimeType": "image/png"}, ...]
    """
    items: List[Dict[str, Any]] = []
    base = Path(images_base_path)
    seen_ids: set = set()

    for image_id in image_refs:
        if not image_id or image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        info = image_index.get(image_id) if image_index else None
        if info:
            file_path = info.get("file_path")
            mime_type = info.get("mime_type", "image/png")
        else:
            # 回退：尝试常见路径 {base}/{collection}/{image_id}.png|.jpg
            file_path = None
            mime_type = "image/png"
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = base / collection_name / f"{image_id}{ext}"
                if candidate.exists():
                    file_path = str(candidate)
                    mime_type = "image/png" if ext == ".png" else "image/jpeg"
                    break

        if not file_path:
            continue

        path = Path(file_path)
        if not path.exists():
            # file_path 可能为相对路径（如 data/images/report/xxx.jpg）
            path = base / file_path
        if not path.exists():
            path = base / collection_name / Path(file_path).name
        if not path.exists():
            logger.debug("图片文件不存在: %s", file_path)
            continue

        try:
            with open(path, "rb") as f:
                raw = f.read()
            b64 = base64.b64encode(raw).decode("ascii")
            items.append({"type": "image", "data": b64, "mimeType": mime_type})
        except Exception as e:
            logger.warning("读取图片失败 %s: %s", path, e)
    return items


def _infer_collection_for_image(
    image_id: str, images_base_path: str
) -> Optional[str]:
    """
    遍历 images 目录下的 collection，在 index.json 中查找 image_id 所属的 collection。
    """
    base = Path(images_base_path)
    if not base.exists():
        return None
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        index_file = sub / "index.json"
        if not index_file.exists():
            continue
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            images = data.get("images", {})
            if isinstance(images, dict) and image_id in images:
                return sub.name
        except Exception:
            continue
    return None


def assemble_content(
    results: List["QueryResult"],
    text_content: str,
    images_base_path: str = "data/images",
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    组装 MCP content 列表：TextContent + ImageContent（当 metadata 含 image_refs 时）。

    Args:
        results: 检索结果
        text_content: 已有的 Markdown 文本（作为第一个 content 项）
        images_base_path: 图片存储根路径
        collection_name: 集合名称，用于定位 index.json；若 None 则按 image_id 推断

    Returns:
        content 列表，如 [{"type":"text","text":...}, {"type":"image","data":...,"mimeType":...}]
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_content}]
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
                coll = _infer_collection_for_image(image_id, images_base_path)
            if not coll:
                continue
            index = _load_image_index(images_base_path, coll)
            items = _image_refs_to_content_items([image_id], coll, images_base_path, index)
            content.extend(items)
    return content


def build_mcp_content_with_images(
    results: List["QueryResult"],
    images_base_path: str = "data/images",
    collection_name: Optional[str] = None,
    max_chars_per_chunk: int = 500,
) -> Dict[str, Any]:
    """
    构建含图片的 MCP content。当 chunk 有 image_refs 时追加 ImageContent。

    内部复用 response_builder 的 markdown 与 citation 逻辑。
    """
    from src.core.response.citation_generator import generate_citations
    from src.core.response.response_builder import _results_to_markdown

    markdown = _results_to_markdown(results, max_chars_per_chunk)
    citations = generate_citations(results)
    content = assemble_content(results, markdown, images_base_path, collection_name)
    return {
        "content": content,
        "structuredContent": {"citations": citations},
        "isError": False,
    }
