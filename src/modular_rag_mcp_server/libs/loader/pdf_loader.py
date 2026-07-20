"""PDF loader facade with focused conversion, image and page helpers."""

from __future__ import annotations

import hashlib
import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast

from modular_rag_mcp_server.ingestion.models import Document
from modular_rag_mcp_server.libs.loader.base_loader import BaseLoader
from modular_rag_mcp_server.libs.loader.pdf_image_extractor import extract_images
from modular_rag_mcp_server.libs.loader.pdf_page_mapper import map_pages, normalize_text
from modular_rag_mcp_server.libs.loader.pdf_placeholder_inserter import (
    append_placeholders,
    insert_placeholders,
)

logger = logging.getLogger(__name__)
ImageMetadata = Dict[str, Any]
PageImage = Tuple[str, float, bytes, ImageMetadata]
PageImages = Dict[int, List[PageImage]]


class _MarkItDownConverter(Protocol):
    def convert(self, path: str) -> Any: ...


class _MarkItDownFactory(Protocol):
    def __call__(self) -> _MarkItDownConverter: ...


class _FitzModule(Protocol):
    def open(self, path: str) -> Any: ...


def _load_optional_attr(module_name: str, attr_name: str) -> Any:
    try:
        return getattr(import_module(module_name), attr_name)
    except (AttributeError, ImportError):
        return None


MarkItDown = cast(Optional[_MarkItDownFactory], _load_optional_attr("markitdown", "MarkItDown"))
MARKITDOWN_AVAILABLE = MarkItDown is not None
fitz = cast(Optional[_FitzModule], _load_optional_attr("fitz", "open"))
if fitz is not None:
    fitz = cast(_FitzModule, import_module("fitz"))
PYMUPDF_AVAILABLE = fitz is not None


def _require_fitz() -> _FitzModule:
    if fitz is None:
        raise RuntimeError("PyMuPDF 未安装。请安装: pip install pymupdf")
    return fitz


class PdfLoader(BaseLoader):
    """Convert PDF files to documents and preserve image references."""

    def __init__(self) -> None:
        converter_factory = MarkItDown
        if not MARKITDOWN_AVAILABLE or converter_factory is None:
            raise RuntimeError("MarkItDown 未安装。请安装: pip install markitdown")
        self._md = converter_factory()
        self._supported_extensions = [".pdf"]

    def load(self, path: str, trace: Optional[Any] = None) -> Document:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        if Path(path).suffix.lower() not in self._supported_extensions:
            raise ValueError(f"不支持的文件类型: {Path(path).suffix}")

        try:
            result = self._md.convert(path)
            markdown_text = getattr(result, "text_content", str(result)) or ""
            doc_id = self._generate_doc_id(path)
            image_data: Dict[str, bytes] = {}
            image_metadata: List[ImageMetadata] = []
            page_images: PageImages = {}
            if PYMUPDF_AVAILABLE:
                try:
                    image_data, image_metadata, page_images = self._extract_images_from_pdf(
                        path, doc_id
                    )
                    markdown_text = self._insert_image_placeholders(
                        markdown_text, page_images, path
                    )
                except Exception as exc:
                    logger.warning("图片提取失败: %s", exc)

            metadata = self._build_metadata(path, result)
            if image_data:
                metadata["image_data"] = image_data
            if image_metadata:
                metadata["images"] = image_metadata
            return Document(id=doc_id, text=markdown_text, metadata=metadata)
        except Exception as exc:
            raise RuntimeError(f"PDF 解析失败: {path}。错误: {exc}") from exc

    def _generate_doc_id(self, path: str) -> str:
        digest = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()
        return f"doc_{digest[:16]}"

    def _build_metadata(self, path: str, result: Any) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "source_path": os.path.abspath(path),
            "doc_type": "pdf",
            "title": Path(path).stem,
            "images": [],
        }
        if getattr(result, "images", None):
            metadata["markitdown_images"] = result.images
        elif getattr(result, "image_refs", None):
            metadata["markitdown_images"] = result.image_refs
        for key, value in (getattr(result, "metadata", None) or {}).items():
            metadata.setdefault(key, value)
        return metadata

    def _extract_images_from_pdf(
        self, pdf_path: str, doc_id: str
    ) -> Tuple[Dict[str, bytes], List[ImageMetadata], PageImages]:
        if not PYMUPDF_AVAILABLE:
            return {}, [], {}
        return extract_images(_require_fitz(), pdf_path, doc_id)

    def _normalize_for_fingerprint_search(self, text: str) -> str:
        return normalize_text(text)

    def _map_pages_to_markdown(
        self, pdf_path: str, markdown_text: str, fingerprint_length: int = 60
    ) -> Dict[int, Tuple[int, int]]:
        if not PYMUPDF_AVAILABLE:
            return {}
        return map_pages(_require_fitz(), pdf_path, markdown_text, fingerprint_length)

    def _insert_image_placeholders_simple(
        self, markdown_text: str, page_images: PageImages
    ) -> str:
        return append_placeholders(markdown_text, page_images)

    def _insert_image_placeholders(
        self,
        markdown_text: str,
        page_images: PageImages,
        pdf_path: str,
        page_to_markdown: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> str:
        if not page_images or not PYMUPDF_AVAILABLE:
            return self._insert_image_placeholders_simple(markdown_text, page_images)
        return insert_placeholders(
            _require_fitz(),
            markdown_text,
            page_images,
            pdf_path,
            page_to_markdown,
            map_pages,
        )

    def get_supported_extensions(self) -> list[str]:
        return self._supported_extensions.copy()
