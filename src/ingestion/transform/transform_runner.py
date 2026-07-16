"""Run ingestion transforms in their stable order."""

import logging
from typing import Any, ContextManager, List, Optional, Protocol

from src.ingestion.models import Chunk

logger = logging.getLogger(__name__)


class ChunkTransformProtocol(Protocol):
    def transform(
        self,
        chunk: Chunk,
        trace: Optional[Any] = None,
    ) -> Chunk: ...


class BoundaryRepairerProtocol(Protocol):
    def repair(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Chunk]: ...


class TransformRunner:
    """Coordinate chunk transforms and optional boundary repair."""

    def __init__(
        self,
        chunk_refiner: ChunkTransformProtocol,
        metadata_enricher: ChunkTransformProtocol,
        image_captioner: ChunkTransformProtocol,
        boundary_repairer: Optional[BoundaryRepairerProtocol] = None,
    ):
        self._chunk_refiner = chunk_refiner
        self._metadata_enricher = metadata_enricher
        self._image_captioner = image_captioner
        self._boundary_repairer = boundary_repairer

    @property
    def chunk_refiner(self) -> ChunkTransformProtocol:
        return self._chunk_refiner

    @chunk_refiner.setter
    def chunk_refiner(self, value: ChunkTransformProtocol) -> None:
        self._chunk_refiner = value

    @property
    def metadata_enricher(self) -> ChunkTransformProtocol:
        return self._metadata_enricher

    @metadata_enricher.setter
    def metadata_enricher(self, value: ChunkTransformProtocol) -> None:
        self._metadata_enricher = value

    @property
    def image_captioner(self) -> ChunkTransformProtocol:
        return self._image_captioner

    @image_captioner.setter
    def image_captioner(self, value: ChunkTransformProtocol) -> None:
        self._image_captioner = value

    @property
    def boundary_repairer(self) -> Optional[BoundaryRepairerProtocol]:
        return self._boundary_repairer

    @boundary_repairer.setter
    def boundary_repairer(
        self,
        value: Optional[BoundaryRepairerProtocol],
    ) -> None:
        self._boundary_repairer = value

    def run(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Chunk]:
        transformed_chunks = []
        for chunk in chunks:
            chunk = self._chunk_refiner.transform(chunk, trace=trace)
            chunk = self._metadata_enricher.transform(chunk, trace=trace)
            chunk = self._image_captioner.transform(chunk, trace=trace)
            transformed_chunks.append(chunk)
        return transformed_chunks

    def repair_boundaries(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
        stage_context: Optional[ContextManager[Any]] = None,
    ) -> List[Chunk]:
        if self._boundary_repairer is None:
            return chunks
        try:
            if stage_context is not None:
                repaired_chunks = chunks
                with stage_context:
                    repaired_chunks = self._boundary_repairer.repair(
                        chunks,
                        trace=trace,
                    )
                return repaired_chunks
            return self._boundary_repairer.repair(chunks, trace=trace)
        except Exception as exc:
            logger.warning("块边界修复失败，跳过: %s", exc)
            return chunks
