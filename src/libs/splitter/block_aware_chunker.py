"""Block-aware chunking for structured parser outputs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.ingestion.models import Chunk
from src.ingestion.parsed_document import ParsedBlock, ParsedDocument
from src.libs.splitter.table_chunker import TableChunker


class BlockAwareChunker:
    """Convert ordered ParsedBlock objects into ingestion Chunk objects."""

    def __init__(self, chunk_size: int = 512, max_table_rows_per_chunk: int = 30) -> None:
        self._chunk_size = max(128, chunk_size)
        self._table_chunker = TableChunker(
            chunk_size=chunk_size,
            max_rows_per_chunk=max_table_rows_per_chunk,
        )

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        pending_text: List[str] = []
        pending_meta: List[Dict[str, Any]] = []
        heading_path: List[str] = []

        def flush_text() -> None:
            nonlocal pending_text, pending_meta
            if not pending_text:
                return
            text = "\n\n".join(part for part in pending_text if part).strip()
            if not text:
                pending_text = []
                pending_meta = []
                return
            metadata = self._base_metadata(document, len(chunks))
            block_ids = [meta.get("block_id") for meta in pending_meta if meta.get("block_id")]
            block_types = [meta.get("block_type") for meta in pending_meta if meta.get("block_type")]
            page_indices = [
                meta.get("page_idx")
                for meta in pending_meta
                if meta.get("page_idx") is not None
            ]
            image_refs: List[str] = []
            for meta in pending_meta:
                image_refs.extend(meta.get("image_refs") or [])
            metadata.update(
                {
                    "block_ids": block_ids,
                    "block_types": block_types,
                    "heading_path": list(heading_path),
                }
            )
            if page_indices:
                metadata["page_indices"] = list(dict.fromkeys(page_indices))
                metadata["page_idx"] = page_indices[0]
            if image_refs:
                metadata["image_refs"] = list(dict.fromkeys(image_refs))
                self._attach_image_assets(metadata, document.metadata, metadata["image_refs"])
            chunks.append(
                Chunk(
                    id=f"{document.id}_chunk_{len(chunks)}",
                    text=text,
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )
            pending_text = []
            pending_meta = []

        for block in document.blocks:
            block_type = (block.type or "text").lower()
            if block_type in {"title", "heading", "header"}:
                level = int(block.metadata.get("heading_level") or block.metadata.get("text_level") or 1)
                level = max(1, min(level, 6))
                heading_path = heading_path[: level - 1]
                if block.text:
                    heading_path.append(block.text.strip())
                self._append_text_block(block, pending_text, pending_meta, heading_path)
                continue

            if block_type == "table":
                flush_text()
                table_chunks = self._table_chunker.chunk_table(
                    block,
                    doc_id=document.id,
                    chunk_index_start=len(chunks),
                    base_metadata=self._base_metadata(document, len(chunks), heading_path),
                    heading_path=heading_path,
                )
                for chunk in table_chunks:
                    if block.image_refs:
                        self._attach_image_assets(chunk.metadata, document.metadata, block.image_refs)
                    chunks.append(chunk)
                continue

            if block_type in {"image", "chart"}:
                self._append_image_block(block, pending_text, pending_meta, heading_path)
            else:
                self._append_text_block(block, pending_text, pending_meta, heading_path)

            if sum(len(part) for part in pending_text) >= self._chunk_size:
                flush_text()

        flush_text()
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = total
        return chunks

    def _base_metadata(
        self,
        document: ParsedDocument,
        chunk_index: int,
        heading_path: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        metadata = {
            k: v for k, v in document.metadata.items()
            if k not in {"image_data", "images"}
        }
        metadata.update(
            {
                "chunk_index": chunk_index,
                "source_doc_id": document.id,
                "heading_path": list(heading_path or []),
            }
        )
        return metadata

    def _append_text_block(
        self,
        block: ParsedBlock,
        pending_text: List[str],
        pending_meta: List[Dict[str, Any]],
        heading_path: List[str],
    ) -> None:
        text = (block.text or "").strip()
        if not text:
            return
        pending_text.append(text)
        pending_meta.append(
            {
                "block_id": block.id,
                "block_type": block.type,
                "page_idx": block.page_idx,
                "bbox": block.bbox,
                "heading_path": list(heading_path),
            }
        )

    def _append_image_block(
        self,
        block: ParsedBlock,
        pending_text: List[str],
        pending_meta: List[Dict[str, Any]],
        heading_path: List[str],
    ) -> None:
        parts: List[str] = []
        if block.text:
            parts.append(block.text.strip())
        parts.extend(f"[IMAGE:{image_id}]" for image_id in block.image_refs)
        if not parts:
            return
        pending_text.append("\n".join(parts))
        pending_meta.append(
            {
                "block_id": block.id,
                "block_type": block.type,
                "page_idx": block.page_idx,
                "bbox": block.bbox,
                "image_refs": block.image_refs,
                "heading_path": list(heading_path),
            }
        )

    def _attach_image_assets(
        self,
        chunk_metadata: Dict[str, Any],
        doc_metadata: Dict[str, Any],
        image_refs: List[str],
    ) -> None:
        image_data_dict = doc_metadata.get("image_data") or {}
        images_list = doc_metadata.get("images") or []
        image_meta_by_id = {
            item.get("image_id"): item
            for item in images_list
            if isinstance(item, dict) and item.get("image_id")
        }
        chunk_image_data = {
            image_id: image_data_dict[image_id]
            for image_id in image_refs
            if image_id in image_data_dict
        }
        chunk_image_metadata = [
            image_meta_by_id[image_id]
            for image_id in image_refs
            if image_id in image_meta_by_id
        ]
        if chunk_image_data:
            chunk_metadata["image_data"] = chunk_image_data
        if chunk_image_metadata:
            chunk_metadata["image_metadata"] = chunk_image_metadata
