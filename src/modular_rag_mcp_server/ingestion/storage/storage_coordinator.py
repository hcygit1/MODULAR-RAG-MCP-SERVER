"""Coordinate document replacement and fallback ingestion storage."""

import logging
from typing import Any, Dict, List, Optional, Protocol

from modular_rag_mcp_server.ingestion.models import Chunk

logger = logging.getLogger(__name__)


class VectorStoreBackendProtocol(Protocol):
    def get_backend(self) -> str: ...


class VectorUpserterProtocol(Protocol):
    def replace_document_chunks(
        self,
        source_doc_id: str,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> int: ...

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> None: ...

    def get_vector_store(self) -> VectorStoreBackendProtocol: ...


class StorageCoordinator:
    """Store encoded chunks while preserving document replacement semantics."""

    def __init__(self, vector_upserter: VectorUpserterProtocol):
        self._vector_upserter = vector_upserter

    @property
    def vector_upserter(self) -> VectorUpserterProtocol:
        return self._vector_upserter

    @vector_upserter.setter
    def vector_upserter(self, value: VectorUpserterProtocol) -> None:
        self._vector_upserter = value

    def store(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> None:
        source_doc_id = ""
        if chunks:
            source_doc_id = str(
                chunks[0].metadata.get("source_doc_id", "")
            ).strip()

        if source_doc_id:
            try:
                self._vector_upserter.replace_document_chunks(
                    source_doc_id=source_doc_id,
                    chunks=chunks,
                    dense_vectors=dense_vectors,
                    sparse_vectors=sparse_vectors,
                    trace=trace,
                    collection_name=collection_name,
                )
                return
            except NotImplementedError:
                logger.warning(
                    "当前 VectorStore 不支持按 source_doc_id 清理，"
                    "回退到普通 upsert。backend=%s",
                    self._vector_upserter.get_vector_store().get_backend(),
                )

        self._vector_upserter.upsert_chunks(
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            trace=trace,
            collection_name=collection_name,
        )
