"""Build ingestion chunks from a document and splitter output."""

import re
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.ingestion.models import Chunk, Document


class SplitterProtocol(Protocol):
    def split_with_metadata(
        self,
        text: str,
        doc_id: str,
        trace: Optional[Any] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]: ...

    def get_strategy(self) -> str: ...

    def get_chunk_size(self) -> int: ...

    def get_chunk_overlap(self) -> int: ...


class DocumentChunkBuilder:
    """Convert splitter output into chunks with stable metadata and offsets."""

    def __init__(self, splitter: SplitterProtocol):
        self._splitter = splitter

    @property
    def splitter(self) -> SplitterProtocol:
        return self._splitter

    @splitter.setter
    def splitter(self, value: SplitterProtocol) -> None:
        self._splitter = value

    def build(
        self,
        document: Document,
        trace: Optional[Any] = None,
    ) -> List[Chunk]:
        if not document or not document.text:
            raise ValueError("Document 或 Document.text 不能为空")

        chunk_tuples = self._splitter.split_with_metadata(
            document.text,
            doc_id=document.id,
            trace=trace,
        )
        image_data = document.metadata.get("image_data") or {}
        image_metadata = {
            item["image_id"]: item
            for item in (document.metadata.get("images") or [])
            if item.get("image_id")
        }

        chunks: List[Chunk] = []
        current_offset = 0
        for index, (chunk_text, split_metadata) in enumerate(chunk_tuples):
            end_offset = current_offset + len(chunk_text)
            metadata = {
                key: value
                for key, value in document.metadata.items()
                if key not in ("image_data", "images")
            }
            metadata.update(
                {
                    "chunk_index": index,
                    "source_doc_id": document.id,
                    "total_chunks": len(chunk_tuples),
                }
            )
            metadata.update(split_metadata)

            image_refs = self.extract_image_refs(chunk_text)
            if image_refs:
                metadata["image_refs"] = image_refs
                chunk_image_data = {
                    image_id: image_data[image_id]
                    for image_id in image_refs
                    if image_id in image_data
                }
                chunk_image_metadata = [
                    image_metadata[image_id]
                    for image_id in image_refs
                    if image_id in image_metadata
                ]
                if chunk_image_data:
                    metadata["image_data"] = chunk_image_data
                if chunk_image_metadata:
                    metadata["image_metadata"] = chunk_image_metadata

            chunks.append(
                Chunk(
                    id=self.generate_chunk_id(document.id, index),
                    text=chunk_text,
                    metadata=metadata,
                    start_offset=current_offset,
                    end_offset=end_offset,
                )
            )
            current_offset = end_offset

        return chunks

    @staticmethod
    def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}_chunk_{chunk_index}"

    @staticmethod
    def extract_image_refs(text: str) -> List[str]:
        return re.findall(r"\[IMAGE:\s*([^\]]+)\]", text)

    def get_splitter_strategy(self) -> str:
        return self._splitter.get_strategy()

    def get_chunk_size(self) -> int:
        return self._splitter.get_chunk_size()

    def get_chunk_overlap(self) -> int:
        return self._splitter.get_chunk_overlap()
