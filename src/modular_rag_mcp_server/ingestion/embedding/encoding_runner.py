"""Stable delegation boundary for chunk encoding."""

from typing import Any, Dict, List, Optional, Protocol, Tuple

from modular_rag_mcp_server.ingestion.models import Chunk


class BatchProcessorProtocol(Protocol):
    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]: ...


class EncodingRunner:
    """Delegate dense and sparse encoding to the batch processor."""

    def __init__(self, batch_processor: BatchProcessorProtocol):
        self._batch_processor = batch_processor

    @property
    def batch_processor(self) -> BatchProcessorProtocol:
        return self._batch_processor

    @batch_processor.setter
    def batch_processor(self, value: BatchProcessorProtocol) -> None:
        self._batch_processor = value

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        return self._batch_processor.process(chunks, trace=trace)
