"""Ingestion-side storage orchestration.

Concrete SQLite/Chroma/Qdrant adapters live in ``libs.vector_store``.  The
classes exported here coordinate ingestion writes and keep the old image
storage API available for callers that still use it.
"""
from modular_rag_mcp_server.ingestion.storage.image_storage import ImageStorage
from modular_rag_mcp_server.ingestion.storage.vector_upserter import VectorUpserter

__all__ = [
    "VectorUpserter",
    "ImageStorage",
]
