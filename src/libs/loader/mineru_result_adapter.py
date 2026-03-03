"""
MinerU 结果适配器

将 MinerURawResult 转为与 PdfLoader 一致的 Document 结构，便于接入 IngestionPipeline。
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.models import Document
from src.libs.loader.mineru_cloud_client import MinerURawResult

logger = logging.getLogger(__name__)


def to_document(raw: MinerURawResult, doc_id: Optional[str] = None) -> Document:
    """
    将 MinerURawResult 转为 Document（与 PdfLoader 结构一致）

    Args:
        raw: MinerU 原始解析结果
        doc_id: 文档 ID，不传则基于 source_path 生成

    Returns:
        Document: 含 [IMAGE: image_id] 占位符、image_data、images
    """
    if doc_id is None or not doc_id.strip():
        doc_id = _generate_doc_id(raw.source_path)

    # 建立 path -> image_id 映射
    path_to_id: Dict[str, str] = {}
    image_data: Dict[str, bytes] = {}
    images_meta: List[Dict[str, Any]] = []

    for idx, (img_path, page_idx, img_bytes) in enumerate(raw.images):
        image_id = f"{doc_id}_page_{page_idx}_img_{idx}"
        path_norm = _normalize_path(img_path)
        path_to_id[img_path] = image_id
        path_to_id[path_norm] = image_id
        path_to_id[path_norm.split("/")[-1]] = image_id  # basename

        image_data[image_id] = img_bytes

        ext = _ext_from_path(img_path)
        mime = _mime_from_ext(ext)
        images_meta.append({
            "image_id": image_id,
            "page": page_idx,
            "y_position": page_idx * 1000,  # 近似，MinerU 无精确 y
            "mime_type": mime,
            "ext": ext,
        })

    # 替换 md 中的 ![](path) 为 [IMAGE: image_id]
    markdown_text = _replace_image_refs(raw.markdown_text, path_to_id)

    metadata: Dict[str, Any] = {
        "source_path": raw.source_path,
        "doc_type": "pdf",
        "title": Path(raw.source_path).stem,
        "image_data": image_data,
        "images": images_meta,
    }

    return Document(id=doc_id, text=markdown_text, metadata=metadata)


def _generate_doc_id(source_path: str) -> str:
    """基于路径生成 doc_id"""
    abs_path = os.path.abspath(source_path)
    h = hashlib.sha256(abs_path.encode("utf-8"))
    return f"doc_{h.hexdigest()[:16]}"


def _normalize_path(p: str) -> str:
    """标准化路径，便于匹配"""
    return p.replace("\\", "/").strip("/")


def _ext_from_path(p: str) -> str:
    """从路径取扩展名"""
    ext = Path(p).suffix.lower()
    return ext.lstrip(".") if ext else "png"


def _mime_from_ext(ext: str) -> str:
    """扩展名 -> mime_type"""
    m = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
    return m.get(ext.lower(), "image/png")


def _replace_image_refs(markdown_text: str, path_to_id: Dict[str, str]) -> str:
    """
    将 md 中的 ![](path) 或 ![alt](path) 替换为 [IMAGE: image_id]
    """
    # 匹配 ![alt](path) 或 ![](path)
    pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    def repl(m: re.Match) -> str:
        alt, path = m.group(1), m.group(2).strip()
        path_norm = _normalize_path(path)
        # 尝试多种匹配：完整 path、含 images/ 前缀、相对路径
        for k, image_id in path_to_id.items():
            if path_norm == k or path_norm.endswith("/" + k) or k in path_norm or path_norm in k:
                return f"[IMAGE: {image_id}]"
        # 若 path 为文件名，尝试匹配 images/xxx 形式
        base = path_norm.split("/")[-1] if "/" in path_norm else path_norm
        for k, image_id in path_to_id.items():
            if k.endswith(base) or base in k:
                return f"[IMAGE: {image_id}]"
        logger.warning("未找到图片映射: %s", path)
        return m.group(0)  # 保留原样

    return pattern.sub(repl, markdown_text)
