"""Regression tests for package ownership and compatibility aliases."""

from pathlib import Path

from modular_rag_mcp_server.ingestion.integrity.file_integrity import (
    FileIntegrityChecker,
)
from modular_rag_mcp_server.ingestion.storage.fts5_bm25_indexer import (
    FTS5BM25Indexer as LegacyFTS5BM25Indexer,
)
from modular_rag_mcp_server.libs.loader.file_integrity import (
    FileIntegrityChecker as LegacyFileIntegrityChecker,
)
from modular_rag_mcp_server.libs.vector_store.fts5_bm25_indexer import (
    FTS5BM25Indexer,
)


def test_legacy_integrity_import_is_the_canonical_implementation() -> None:
    assert FileIntegrityChecker is LegacyFileIntegrityChecker
    assert FileIntegrityChecker.__module__.startswith(
        "modular_rag_mcp_server.ingestion.integrity"
    )


def test_legacy_fts_import_is_the_canonical_vector_store_adapter() -> None:
    assert FTS5BM25Indexer is LegacyFTS5BM25Indexer
    assert FTS5BM25Indexer.__module__.startswith(
        "modular_rag_mcp_server.libs.vector_store"
    )


def test_sparse_retriever_does_not_import_ingestion_storage() -> None:
    source = Path(
        "src/modular_rag_mcp_server/core/query_engine/sparse_retriever.py"
    ).read_text(encoding="utf-8")

    assert "modular_rag_mcp_server.ingestion.storage" not in source
