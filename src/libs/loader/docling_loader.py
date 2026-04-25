"""Docling loader that produces ingestion chunks directly."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion.models import Chunk
from src.libs.loader.mineru_result_adapter import _generate_doc_id


class DoclingLoader:
    """Use Docling DocumentConverter + HybridChunker as a structured frontend."""

    def __init__(self, chunk_size: int = 512) -> None:
        self._chunk_size = chunk_size

    def load_chunks(self, path: str, doc_id: Optional[str] = None) -> List[Chunk]:
        try:
            from docling.document_converter import DocumentConverter
            from docling.chunking import HybridChunker
        except ImportError as exc:
            raise RuntimeError(
                "Docling 未安装，请先安装可选依赖: pip install 'docling>=2.0.0'"
            ) from exc

        source = Path(path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        if doc_id is None or not doc_id.strip():
            doc_id = _generate_doc_id(str(source.resolve()))

        converter = DocumentConverter()
        result = converter.convert(str(source))
        dl_doc = result.document

        chunker = HybridChunker()
        try:
            chunk_iter = chunker.chunk(dl_doc=dl_doc)
        except TypeError:
            chunk_iter = chunker.chunk(dl_doc)

        chunks: List[Chunk] = []
        for idx, dl_chunk in enumerate(chunk_iter):
            text = self._contextualize(chunker, dl_chunk)
            if not text:
                continue
            metadata = self._metadata_from_chunk(
                dl_chunk,
                doc_id=doc_id,
                path=str(source.resolve()),
                chunk_index=len(chunks),
            )
            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{len(chunks)}",
                    text=text,
                    metadata=metadata,
                )
            )

        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = total
        return chunks

    def _contextualize(self, chunker: Any, chunk: Any) -> str:
        for call in (
            lambda: chunker.contextualize(chunk=chunk),
            lambda: chunker.contextualize(chunk),
        ):
            try:
                text = call()
                if text:
                    return str(text).strip()
            except TypeError:
                continue

        text = getattr(chunk, "text", None)
        if text:
            return str(text).strip()
        return str(chunk).strip()

    def _metadata_from_chunk(
        self,
        chunk: Any,
        *,
        doc_id: str,
        path: str,
        chunk_index: int,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "source_path": os.path.abspath(path),
            "doc_type": Path(path).suffix.lower().lstrip(".") or "document",
            "title": Path(path).stem,
            "parser": "docling",
            "chunk_index": chunk_index,
            "source_doc_id": doc_id,
        }

        meta = getattr(chunk, "meta", None)
        if meta is not None:
            headings = getattr(meta, "headings", None)
            if headings:
                metadata["heading_path"] = [str(item) for item in headings]

            doc_items = getattr(meta, "doc_items", None)
            if doc_items:
                metadata["docling_item_count"] = len(doc_items)
                item_labels = []
                page_numbers = []
                for item in doc_items:
                    label = getattr(item, "label", None)
                    if label:
                        item_labels.append(str(label))
                    prov = getattr(item, "prov", None)
                    if prov:
                        for prov_item in prov:
                            page_no = getattr(prov_item, "page_no", None)
                            if page_no is not None:
                                page_numbers.append(page_no)
                if item_labels:
                    metadata["block_types"] = list(dict.fromkeys(item_labels))
                if page_numbers:
                    metadata["page_idx"] = min(page_numbers)

        return metadata
