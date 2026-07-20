"""Build the replaceable components used by the ingestion pipeline.

Keeping dependency construction here leaves :class:`IngestionPipeline` focused
on executing the ingestion stages and preserves one place for default wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from modular_rag_mcp_server.core.settings import Settings
from modular_rag_mcp_server.ingestion.document_chunk_builder import DocumentChunkBuilder
from modular_rag_mcp_server.ingestion.embedding.batch_processor import BatchProcessor
from modular_rag_mcp_server.ingestion.embedding.dense_encoder import DenseEncoder
from modular_rag_mcp_server.ingestion.embedding.encoding_runner import EncodingRunner
from modular_rag_mcp_server.ingestion.embedding.sparse_encoder import SparseEncoder
from modular_rag_mcp_server.ingestion.integrity import FileIntegrityChecker
from modular_rag_mcp_server.ingestion.storage.storage_coordinator import StorageCoordinator
from modular_rag_mcp_server.ingestion.storage.vector_upserter import VectorUpserter
from modular_rag_mcp_server.ingestion.transform.boundary_repairer import BoundaryRepairer
from modular_rag_mcp_server.ingestion.transform.chunk_refiner import ChunkRefiner
from modular_rag_mcp_server.ingestion.transform.image_captioner import ImageCaptioner
from modular_rag_mcp_server.ingestion.transform.metadata_enricher import MetadataEnricher
from modular_rag_mcp_server.ingestion.transform.transform_runner import TransformRunner
from modular_rag_mcp_server.libs.embedding.embedding_factory import EmbeddingFactory
from modular_rag_mcp_server.libs.llm.llm_factory import LLMFactory
from modular_rag_mcp_server.libs.splitter.splitter_factory import SplitterFactory
from modular_rag_mcp_server.libs.vector_store.vector_store_factory import VectorStoreFactory


@dataclass
class PipelineComponents:
    """All collaborators required to execute a document ingestion request."""

    integrity_checker: FileIntegrityChecker
    chunk_builder: DocumentChunkBuilder
    transform_runner: TransformRunner
    encoding_runner: EncodingRunner
    storage_coordinator: StorageCoordinator


def resolve_integrity_checker(
    integrity_checker: Optional[FileIntegrityChecker],
) -> FileIntegrityChecker:
    """Preserve an injected checker even when its truth value is false."""
    return (
        integrity_checker
        if integrity_checker is not None
        else FileIntegrityChecker()
    )


class PipelineComponentFactory:
    """Create the default ingestion component graph from application settings."""

    @staticmethod
    def create(
        settings: Settings,
        *,
        splitter_strategy: Optional[str] = None,
        llm: Optional[Any] = None,
        vision_llm: Optional[Any] = None,
        embedding: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        integrity_checker: Optional[FileIntegrityChecker] = None,
    ) -> PipelineComponents:
        ingestion_config = settings.ingestion
        checker = resolve_integrity_checker(integrity_checker)

        splitter = SplitterFactory.create(settings, strategy=splitter_strategy)
        chunk_builder = DocumentChunkBuilder(splitter)

        if llm is None and (
            ingestion_config.enable_llm_refinement
            or ingestion_config.enable_metadata_enrichment
        ):
            try:
                llm = LLMFactory.create(config=settings.llm)
            except (ValueError, NotImplementedError):
                llm = None

        chunk_refiner = ChunkRefiner(config=ingestion_config, llm=llm)
        metadata_enricher = MetadataEnricher(config=ingestion_config, llm=llm)

        if vision_llm is None and ingestion_config.enable_image_captioning:
            vision_llm = LLMFactory.create_vision_llm(settings)

        image_captioner = ImageCaptioner(
            config=ingestion_config,
            vision_llm_config=settings.vision_llm,
            vision_llm=vision_llm,
            image_storage=None,
        )

        boundary_repairer = None
        if ingestion_config.enable_boundary_repair:
            boundary_repairer = BoundaryRepairer(
                config=ingestion_config,
                llm=llm,
            )
        transform_runner = TransformRunner(
            chunk_refiner=chunk_refiner,
            metadata_enricher=metadata_enricher,
            image_captioner=image_captioner,
            boundary_repairer=boundary_repairer,
        )

        if embedding is None:
            embedding = EmbeddingFactory.create(settings)
        dense_encoder = DenseEncoder(embedding=embedding, batch_size=None)
        batch_processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=SparseEncoder(),
            batch_size=ingestion_config.batch_size,
        )
        encoding_runner = EncodingRunner(batch_processor)

        if vector_store is None:
            vector_store = VectorStoreFactory.create(settings)
        storage_coordinator = StorageCoordinator(VectorUpserter(vector_store))

        return PipelineComponents(
            integrity_checker=checker,
            chunk_builder=chunk_builder,
            transform_runner=transform_runner,
            encoding_runner=encoding_runner,
            storage_coordinator=storage_coordinator,
        )
