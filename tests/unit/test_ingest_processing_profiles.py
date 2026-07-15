"""入库入口 processing profile 契约测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.loader.file_integrity import build_processing_profile
from src.libs.vector_store.fake_vector_store import FakeVectorStore
from src.mcp_server.tools.ingest_document_docling import (
    execute_ingest_document_docling,
)
from src.mcp_server.tools.ingest_document_mineru import (
    execute_ingest_document_mineru,
)
from src.mcp_server.tools.ingest_document_normal import (
    execute_ingest_document_normal,
)


@pytest.mark.parametrize(
    ("suffix", "loader_name"),
    [
        (".md", "markdown"),
        (".txt", "text"),
    ],
)
@patch(
    "src.mcp_server.tools.ingest_document_normal.build_processing_profile",
    return_value="profile-normal",
)
@patch("src.mcp_server.tools.ingest_document_normal.load_mcp_settings")
@patch("src.mcp_server.tools.ingest_document_normal.FileIntegrityChecker")
def test_normal_integrity_uses_actual_loader_type(
    mock_integrity_cls: MagicMock,
    mock_load_settings: MagicMock,
    mock_build_profile: MagicMock,
    suffix: str,
    loader_name: str,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / f"document{suffix}"
    file_path.write_text("same content", encoding="utf-8")
    settings = MagicMock()
    settings.vector_store.collection_name = "docs"
    mock_load_settings.return_value = settings
    checker = mock_integrity_cls.return_value
    checker.compute_sha256.return_value = "shared-hash"
    checker.should_skip.return_value = True

    result = execute_ingest_document_normal({"file_path": str(file_path)})

    assert result["structuredContent"]["skipped"] is True
    mock_build_profile.assert_called_once_with(
        settings,
        loader_name=loader_name,
    )


@patch(
    "src.mcp_server.tools.ingest_document_docling.build_processing_profile",
    return_value="profile-docling",
    create=True,
)
@patch("src.mcp_server.tools.ingest_document_docling.load_mcp_settings")
@patch("src.mcp_server.tools.ingest_document_docling.FileIntegrityChecker")
def test_docling_integrity_is_scoped_by_processing_profile(
    mock_integrity_cls: MagicMock,
    mock_load_settings: MagicMock,
    mock_build_profile: MagicMock,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake")
    settings = MagicMock()
    settings.vector_store.collection_name = "docs"
    mock_load_settings.return_value = settings
    checker = mock_integrity_cls.return_value
    checker.compute_sha256.return_value = "hash-docling"
    checker.should_skip.return_value = True

    result = execute_ingest_document_docling({"file_path": str(file_path)})

    assert result["structuredContent"]["skipped"] is True
    mock_build_profile.assert_called_once_with(settings, loader_name="docling")
    checker.should_skip.assert_called_once_with(
        "hash-docling",
        "docs",
        "profile-docling",
    )


@patch(
    "src.mcp_server.tools.ingest_document_mineru.build_processing_profile",
    return_value="profile-mineru",
    create=True,
)
@patch("src.mcp_server.tools.ingest_document_mineru.load_mcp_settings")
@patch("src.mcp_server.tools.ingest_document_mineru.FileIntegrityChecker")
def test_mineru_integrity_is_scoped_by_processing_profile(
    mock_integrity_cls: MagicMock,
    mock_load_settings: MagicMock,
    mock_build_profile: MagicMock,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake")
    settings = MagicMock()
    settings.vector_store.collection_name = "docs"
    mock_load_settings.return_value = settings
    checker = mock_integrity_cls.return_value
    checker.compute_sha256.return_value = "hash-mineru"
    checker.should_skip.return_value = True

    result = execute_ingest_document_mineru({"file_path": str(file_path)})

    assert result["structuredContent"]["skipped"] is True
    mock_build_profile.assert_called_once_with(settings, loader_name="mineru")
    checker.should_skip.assert_called_once_with(
        "hash-mineru",
        "docs",
        "profile-mineru",
    )


def test_pipeline_integrity_is_scoped_by_pdf_processing_profile(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake")
    settings = load_settings("config/settings.yaml")
    settings.ingestion.enable_llm_refinement = False
    settings.ingestion.enable_metadata_enrichment = False
    settings.ingestion.enable_image_captioning = False
    checker = MagicMock()
    checker.compute_sha256.return_value = "hash-pipeline"
    checker.should_skip.return_value = True
    pipeline = IngestionPipeline(
        settings,
        embedding=FakeEmbedding(dimension=settings.vector_store.embedding_dim),
        vector_store=FakeVectorStore(),
        integrity_checker=checker,
    )

    pipeline.process(str(file_path), "docs")

    expected_profile = build_processing_profile(settings, loader_name="pdf")
    checker.should_skip.assert_called_once_with(
        "hash-pipeline",
        "docs",
        expected_profile,
    )
