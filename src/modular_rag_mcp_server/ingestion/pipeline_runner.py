"""Execute the transform, encoding and storage stages of ingestion."""

from __future__ import annotations

from typing import Any, List, Optional

from modular_rag_mcp_server.core.trace.trace_context import TraceContext
from modular_rag_mcp_server.ingestion.document_chunk_builder import DocumentChunkBuilder
from modular_rag_mcp_server.ingestion.embedding.encoding_runner import EncodingRunner
from modular_rag_mcp_server.ingestion.models import Chunk, Document
from modular_rag_mcp_server.ingestion.storage.storage_coordinator import StorageCoordinator
from modular_rag_mcp_server.ingestion.transform.transform_runner import TransformRunner


class PipelineRunner:
    """Run the stages after a document has been loaded and before it is marked complete."""

    def __init__(
        self,
        *,
        chunk_builder: Optional[DocumentChunkBuilder],
        transform_runner: TransformRunner,
        encoding_runner: EncodingRunner,
        storage_coordinator: StorageCoordinator,
    ) -> None:
        self._chunk_builder = chunk_builder
        self._transform_runner = transform_runner
        self._encoding_runner = encoding_runner
        self._storage_coordinator = storage_coordinator

    def process_document(
        self,
        document: Document,
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> int:
        if not document or not document.text:
            raise ValueError("Document 或 Document.text 不能为空")
        if not collection_name or not str(collection_name).strip():
            raise ValueError("collection_name 不能为空")

        if trace is None:
            trace = TraceContext(operation="ingestion")
        stage_trace = trace if isinstance(trace, TraceContext) else None
        if stage_trace:
            stage_trace.set_metric("doc_id", document.id)
            stage_trace.set_metric("collection_name", collection_name)

        try:
            if stage_trace:
                with stage_trace.stage("split"):
                    chunks = self.split_document(document, trace=trace)
                stage_trace.set_metric("chunk_count", len(chunks))
            else:
                chunks = self.split_document(document, trace=trace)
            if not chunks:
                raise RuntimeError("文档切分后未产生任何 chunks")
        except Exception as exc:
            raise RuntimeError(f"文档切分失败: {exc}") from exc

        return self.process_chunks(chunks, collection_name, trace=trace)

    def process_chunks(
        self,
        chunks: List[Chunk],
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> int:
        if not chunks:
            raise ValueError("chunks 不能为空")
        if not collection_name or not str(collection_name).strip():
            raise ValueError("collection_name 不能为空")

        if trace is None:
            trace = TraceContext(operation="ingestion")
        stage_trace = trace if isinstance(trace, TraceContext) else None
        if stage_trace:
            stage_trace.set_metric("collection_name", collection_name)
            stage_trace.set_metric("chunk_count", len(chunks))
            source_doc_id = str(chunks[0].metadata.get("source_doc_id", "")).strip()
            if source_doc_id:
                stage_trace.set_metric("doc_id", source_doc_id)

        try:
            if stage_trace:
                with stage_trace.stage("transform"):
                    transformed_chunks = self._apply_transforms(chunks, trace=trace)
            else:
                transformed_chunks = self._apply_transforms(chunks, trace=trace)
        except Exception as exc:
            raise RuntimeError(f"Chunk 转换失败: {exc}") from exc

        boundary_repairer = self._transform_runner.boundary_repairer
        if boundary_repairer is not None:
            stage_context = stage_trace.stage("boundary_repair") if stage_trace else None
            transformed_chunks = self._transform_runner.repair_boundaries(
                transformed_chunks,
                trace=trace,
                stage_context=stage_context,
            )

        try:
            if stage_trace:
                with stage_trace.stage("encode"):
                    dense_vectors, sparse_vectors = self._encoding_runner.encode(
                        transformed_chunks,
                        trace=trace,
                    )
            else:
                dense_vectors, sparse_vectors = self._encoding_runner.encode(
                    transformed_chunks,
                    trace=trace,
                )
        except Exception as exc:
            raise RuntimeError(f"向量编码失败: {exc}") from exc

        try:
            if stage_trace:
                with stage_trace.stage("store"):
                    self._store_results(
                        transformed_chunks,
                        dense_vectors,
                        sparse_vectors,
                        collection_name,
                        trace=trace,
                    )
            else:
                self._store_results(
                    transformed_chunks,
                    dense_vectors,
                    sparse_vectors,
                    collection_name,
                    trace=trace,
                )
        except Exception as exc:
            raise RuntimeError(f"存储失败: {exc}") from exc

        return len(transformed_chunks)

    def _apply_transforms(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Chunk]:
        return self._transform_runner.run(chunks, trace=trace)

    def _store_results(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[dict[str, float]],
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> None:
        self._storage_coordinator.store(
            chunks,
            dense_vectors,
            sparse_vectors,
            collection_name,
            trace=trace,
        )

    def split_document(
        self,
        document: Document,
        trace: Optional[Any] = None,
    ) -> List[Chunk]:
        if self._chunk_builder is None:
            raise RuntimeError("chunk builder 未初始化")
        return self._chunk_builder.build(document, trace=trace)

    def generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        if self._chunk_builder is None:
            raise RuntimeError("chunk builder 未初始化")
        return self._chunk_builder.generate_chunk_id(doc_id, chunk_index)

    def extract_image_refs(self, text: str) -> List[str]:
        if self._chunk_builder is None:
            raise RuntimeError("chunk builder 未初始化")
        return self._chunk_builder.extract_image_refs(text)

    def get_splitter_strategy(self) -> str:
        if self._chunk_builder is None:
            raise RuntimeError("chunk builder 未初始化")
        return self._chunk_builder.get_splitter_strategy()

    def get_chunk_size(self) -> int:
        if self._chunk_builder is None:
            raise RuntimeError("chunk builder 未初始化")
        return self._chunk_builder.get_chunk_size()

    def get_chunk_overlap(self) -> int:
        if self._chunk_builder is None:
            raise RuntimeError("chunk builder 未初始化")
        return self._chunk_builder.get_chunk_overlap()
