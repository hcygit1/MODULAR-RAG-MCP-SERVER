"""MinerU structured result adapter."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.ingestion.models import Chunk
from src.libs.loader.mineru_cloud_client import MinerURawResult
from src.libs.loader.mineru_result_adapter import (
    _ext_from_path,
    _generate_doc_id,
    _mime_from_ext,
    _normalize_path,
)
from src.ingestion.parsed_document import ParsedBlock, ParsedDocument
from src.libs.splitter.block_aware_chunker import BlockAwareChunker

logger = logging.getLogger(__name__)


def to_chunks(
    raw: MinerURawResult,
    *,
    doc_id: Optional[str] = None,
    chunk_size: int = 512,
    max_table_rows_per_chunk: int = 30,
) -> List[Chunk]:
    """Convert MinerU content_list blocks directly to ingestion chunks."""
    parsed = MinerUBlockParser().parse(raw, doc_id=doc_id)
    return BlockAwareChunker(
        chunk_size=chunk_size,
        max_table_rows_per_chunk=max_table_rows_per_chunk,
    ).chunk(parsed)


class MinerUBlockParser:
    """Parse MinerU content_list into ordered blocks with table/image metadata."""

    def parse(self, raw: MinerURawResult, doc_id: Optional[str] = None) -> ParsedDocument:
        if doc_id is None or not doc_id.strip():
            doc_id = _generate_doc_id(raw.source_path)

        path_to_id, image_data, images_meta = self._build_image_maps(raw, doc_id)
        metadata: Dict[str, Any] = {
            "source_path": raw.source_path,
            "doc_type": "pdf",
            "title": Path(raw.source_path).stem,
            "parser": "mineru",
            "image_data": image_data,
            "images": images_meta,
        }

        items = list(self._iter_content_items(raw.content_list or []))
        if not items:
            return ParsedDocument(
                id=doc_id,
                blocks=[
                    ParsedBlock(
                        id=f"{doc_id}_block_0",
                        type="text",
                        text=raw.markdown_text,
                        metadata={"parser_fallback": "markdown"},
                    )
                ],
                metadata=metadata,
            )

        blocks: List[ParsedBlock] = []
        for idx, item in enumerate(items):
            block = self._parse_item(
                item,
                doc_id=doc_id,
                index=idx,
                path_to_id=path_to_id,
            )
            if block:
                blocks.append(block)

        return ParsedDocument(id=doc_id, blocks=blocks, metadata=metadata)

    def _build_image_maps(
        self,
        raw: MinerURawResult,
        doc_id: str,
    ) -> Tuple[Dict[str, str], Dict[str, bytes], List[Dict[str, Any]]]:
        path_to_id: Dict[str, str] = {}
        image_data: Dict[str, bytes] = {}
        images_meta: List[Dict[str, Any]] = []

        for idx, (img_path, page_idx, img_bytes) in enumerate(raw.images):
            image_id = f"{doc_id}_page_{page_idx}_img_{idx}"
            path_norm = _normalize_path(img_path)
            basename = path_norm.split("/")[-1]
            path_to_id[img_path] = image_id
            path_to_id[path_norm] = image_id
            path_to_id[basename] = image_id
            image_data[image_id] = img_bytes

            ext = _ext_from_path(img_path)
            images_meta.append(
                {
                    "image_id": image_id,
                    "page": page_idx,
                    "page_idx": page_idx,
                    "y_position": page_idx * 1000,
                    "mime_type": _mime_from_ext(ext),
                    "ext": ext,
                    "source_path": img_path,
                }
            )

        return path_to_id, image_data, images_meta

    def _iter_content_items(self, content_list: Iterable[Any]) -> Iterable[Dict[str, Any]]:
        for item in content_list:
            if isinstance(item, dict):
                if isinstance(item.get("content"), list):
                    yield from self._iter_content_items(item["content"])
                else:
                    yield item
            elif isinstance(item, list):
                yield from self._iter_content_items(item)

    def _parse_item(
        self,
        item: Dict[str, Any],
        *,
        doc_id: str,
        index: int,
        path_to_id: Dict[str, str],
    ) -> Optional[ParsedBlock]:
        raw_type = str(item.get("type") or "text").lower()
        block_id = f"{doc_id}_block_{index}"
        page_idx = item.get("page_idx")
        bbox = item.get("bbox")

        if raw_type == "table":
            image_refs = self._match_image_refs(item, path_to_id)
            table_html = item.get("table_body") or item.get("html")
            if not isinstance(table_html, str):
                content = item.get("content")
                table_html = content if isinstance(content, str) else None
            return ParsedBlock(
                id=block_id,
                type="table",
                text=self._text_from_item(item),
                html=table_html,
                image_refs=image_refs,
                page_idx=page_idx,
                bbox=bbox,
                metadata={
                    "table_id": block_id,
                    "table_caption": item.get("table_caption") or item.get("caption"),
                    "table_footnote": item.get("table_footnote"),
                    "raw_type": raw_type,
                    "img_path": item.get("img_path") or item.get("image_path"),
                },
            )

        if raw_type in {"image", "chart"}:
            image_refs = self._match_image_refs(item, path_to_id)
            caption = item.get("image_caption") or item.get("caption") or item.get("img_caption")
            text_parts = []
            if caption:
                text_parts.append(self._stringify(caption))
            text = self._text_from_item(item)
            if text:
                text_parts.append(text)
            return ParsedBlock(
                id=block_id,
                type=raw_type,
                text="\n".join(part for part in text_parts if part),
                image_refs=image_refs,
                page_idx=page_idx,
                bbox=bbox,
                metadata={
                    "raw_type": raw_type,
                    "caption": caption,
                    "img_path": item.get("img_path") or item.get("image_path"),
                },
            )

        if raw_type in {"title", "header"} or item.get("text_level"):
            return ParsedBlock(
                id=block_id,
                type="title",
                text=self._text_from_item(item),
                page_idx=page_idx,
                bbox=bbox,
                metadata={
                    "raw_type": raw_type,
                    "heading_level": item.get("text_level") or item.get("level") or 1,
                },
            )

        if raw_type in {"equation", "code", "list", "text", "paragraph", "footer", "page_footnote"}:
            text = self._text_from_item(item)
            if not text:
                return None
            return ParsedBlock(
                id=block_id,
                type=raw_type,
                text=text,
                page_idx=page_idx,
                bbox=bbox,
                metadata={"raw_type": raw_type},
            )

        text = self._text_from_item(item)
        if not text:
            logger.debug("跳过无法转为文本的 MinerU block: type=%s keys=%s", raw_type, list(item.keys()))
            return None
        return ParsedBlock(
            id=block_id,
            type="text",
            text=text,
            page_idx=page_idx,
            bbox=bbox,
            metadata={"raw_type": raw_type},
        )

    def _match_image_refs(self, item: Dict[str, Any], path_to_id: Dict[str, str]) -> List[str]:
        refs: List[str] = []
        for field in ("img_path", "image_path", "path"):
            path = item.get(field)
            if not path:
                continue
            image_id = self._lookup_image_id(str(path), path_to_id)
            if image_id:
                refs.append(image_id)
        return list(dict.fromkeys(refs))

    def _lookup_image_id(self, path: str, path_to_id: Dict[str, str]) -> Optional[str]:
        path_norm = _normalize_path(path)
        basename = path_norm.split("/")[-1]
        for key in (path, path_norm, basename):
            if key in path_to_id:
                return path_to_id[key]
        for key, image_id in path_to_id.items():
            if path_norm == key or path_norm.endswith("/" + key) or key in path_norm or path_norm in key:
                return image_id
        return None

    def _text_from_item(self, item: Dict[str, Any]) -> str:
        for key in ("text", "content", "table_text"):
            value = item.get(key)
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                return self._stringify(value)
        return ""

    def _stringify(self, value: Any) -> str:
        if isinstance(value, list):
            return " ".join(self._stringify(item) for item in value if item)
        if isinstance(value, dict):
            for key in ("text", "content", "caption"):
                if key in value:
                    return self._stringify(value[key])
            return " ".join(self._stringify(item) for item in value.values() if item)
        return str(value).strip()
