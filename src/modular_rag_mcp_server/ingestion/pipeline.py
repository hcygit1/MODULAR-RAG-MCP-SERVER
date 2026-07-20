"""Document ingestion facade.

The facade owns integrity checks and PDF loading.  Stage execution lives in
:class:`PipelineRunner`, while component construction lives in
:class:`PipelineComponentFactory`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from modular_rag_mcp_server.core.settings import Settings
from modular_rag_mcp_server.core.trace.trace_context import TraceContext
from modular_rag_mcp_server.ingestion.integrity import (
    FileIntegrityChecker,
    build_processing_profile,
)
from modular_rag_mcp_server.ingestion.models import Chunk, Document
from modular_rag_mcp_server.ingestion.pipeline_components import PipelineComponentFactory
from modular_rag_mcp_server.ingestion.pipeline_runner import PipelineRunner
from modular_rag_mcp_server.libs.loader.pdf_loader import PdfLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Public ingestion facade kept compatible with existing callers."""

    def __init__(
        self,
        settings: Settings,
        splitter_strategy: Optional[str] = None,
        llm: Optional[Any] = None,
        vision_llm: Optional[Any] = None,
        embedding: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        integrity_checker: Optional[FileIntegrityChecker] = None,
    ):
        self._settings = settings
        self._ingestion_config = settings.ingestion
        components = PipelineComponentFactory.create(
            settings,
            splitter_strategy=splitter_strategy,
            llm=llm,
            vision_llm=vision_llm,
            embedding=embedding,
            vector_store=vector_store,
            integrity_checker=integrity_checker,
        )
        self._integrity_checker = components.integrity_checker
        self._chunk_builder = components.chunk_builder
        self._transform_runner = components.transform_runner
        self._encoding_runner = components.encoding_runner
        self._storage_coordinator = components.storage_coordinator

    # These properties preserve the old injection points while the runner uses
    # the current component objects on every call.
    @property
    def _splitter(self) -> Any:
        return self._chunk_builder.splitter

    @_splitter.setter
    def _splitter(self, value: Any) -> None:
        self._chunk_builder.splitter = value

    @property
    def _chunk_refiner(self) -> Any:
        return self._transform_runner.chunk_refiner

    @_chunk_refiner.setter
    def _chunk_refiner(self, value: Any) -> None:
        self._transform_runner.chunk_refiner = value

    @property
    def _metadata_enricher(self) -> Any:
        return self._transform_runner.metadata_enricher

    @_metadata_enricher.setter
    def _metadata_enricher(self, value: Any) -> None:
        self._transform_runner.metadata_enricher = value

    @property
    def _image_captioner(self) -> Any:
        return self._transform_runner.image_captioner

    @_image_captioner.setter
    def _image_captioner(self, value: Any) -> None:
        self._transform_runner.image_captioner = value

    @property
    def _boundary_repairer(self) -> Optional[Any]:
        return self._transform_runner.boundary_repairer

    @_boundary_repairer.setter
    def _boundary_repairer(self, value: Optional[Any]) -> None:
        self._transform_runner.boundary_repairer = value

    @property
    def _batch_processor(self) -> Any:
        return self._encoding_runner.batch_processor

    @_batch_processor.setter
    def _batch_processor(self, value: Any) -> None:
        self._encoding_runner.batch_processor = value

    @property
    def _vector_upserter(self) -> Any:
        return self._storage_coordinator.vector_upserter

    @_vector_upserter.setter
    def _vector_upserter(self, value: Any) -> None:
        self._storage_coordinator.vector_upserter = value

    def _runner(self) -> PipelineRunner:
        return PipelineRunner(
            chunk_builder=getattr(self, "_chunk_builder", None),
            transform_runner=self._transform_runner,
            encoding_runner=self._encoding_runner,
            storage_coordinator=self._storage_coordinator,
        )

    def close(self) -> None:
        """Close resources owned by the configured vector store."""
        store = self._vector_upserter.get_vector_store()
        if store is not None:
            store.close()

    def process(
        self,
        file_path: str,
        collection_name: str,
        trace: Optional[Any] = None,
        force: bool = False,
    ) -> None:
        """Load one PDF, run ingestion and record successful processing."""
        if not file_path:
            raise ValueError("file_path 不能为空")
        if not collection_name:
            raise ValueError("collection_name 不能为空")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if trace is None:
            trace = TraceContext(operation="ingestion")
        stage_trace = trace if isinstance(trace, TraceContext) else None
        if stage_trace:
            stage_trace.set_metric("file_path", file_path)
            stage_trace.set_metric("collection_name", collection_name)

        try:
            processing_profile = build_processing_profile(
                self._settings,
                loader_name="pdf",
            )
            if stage_trace:
                with stage_trace.stage("integrity_check"):
                    file_hash = self._integrity_checker.compute_sha256(file_path)
                    should_skip = (
                        False
                        if force
                        else self._integrity_checker.should_skip(
                            file_hash,
                            collection_name,
                            processing_profile,
                        )
                    )
            else:
                file_hash = self._integrity_checker.compute_sha256(file_path)
                should_skip = (
                    False
                    if force
                    else self._integrity_checker.should_skip(
                        file_hash,
                        collection_name,
                        processing_profile,
                    )
                )
            if should_skip:
                if stage_trace:
                    stage_trace.set_metric("skipped", True)
                return
        except Exception as exc:
            raise RuntimeError(f"文件完整性检查失败: {exc}") from exc

        try:
            if stage_trace:
                with stage_trace.stage("load"):
                    document = PdfLoader().load(file_path, trace=trace)
            else:
                document = PdfLoader().load(file_path, trace=trace)
        except Exception as exc:
            raise RuntimeError(f"文档加载失败: {exc}") from exc

        self.process_document(document, collection_name, trace=trace)

        try:
            self._integrity_checker.mark_success(
                file_hash,
                collection_name,
                processing_profile,
            )
        except Exception as exc:
            logger.warning(
                "标记文件处理成功失败 (file_hash=%s)，可能导致后续重复处理: %s",
                file_hash[:16] if file_hash else "?",
                exc,
            )

        if stage_trace:
            stage_trace.set_metric("skipped", False)

    def process_document(
        self,
        document: Document,
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> int:
        return self._runner().process_document(document, collection_name, trace=trace)

    def process_chunks(
        self,
        chunks: List[Chunk],
        collection_name: str,
        trace: Optional[Any] = None,
    ) -> int:
        return self._runner().process_chunks(chunks, collection_name, trace=trace)

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
        sparse_vectors: List[Dict[str, float]],
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
        return self._chunk_builder.build(document, trace=trace)

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        return self._chunk_builder.generate_chunk_id(doc_id, chunk_index)

    def _extract_image_refs(self, text: str) -> List[str]:
        return self._chunk_builder.extract_image_refs(text)

    def get_splitter_strategy(self) -> str:
        return self._chunk_builder.get_splitter_strategy()

    def get_chunk_size(self) -> int:
        return self._chunk_builder.get_chunk_size()

    def get_chunk_overlap(self) -> int:
        return self._chunk_builder.get_chunk_overlap()
