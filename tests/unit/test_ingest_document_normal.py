"""ingest_document_normal 单元测试"""
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from src.ingestion.models import Document
from src.mcp_server.tools.ingest_document_normal import execute_ingest_document_normal


def test_execute_file_path_empty() -> None:
    """file_path 为空时返回 INVALID_PARAMS"""
    result = execute_ingest_document_normal({"file_path": ""})
    assert result["isError"] is True
    assert result["structuredContent"]["errorType"] == "INVALID_PARAMS"
    assert result["structuredContent"]["chunk_count"] == 0


def test_execute_file_path_none() -> None:
    """file_path 为 None 时返回 INVALID_PARAMS"""
    result = execute_ingest_document_normal({"file_path": None})
    assert result["isError"] is True
    assert result["structuredContent"]["errorType"] == "INVALID_PARAMS"


def test_execute_file_not_found() -> None:
    """文件不存在时返回 RESOURCE_NOT_FOUND"""
    result = execute_ingest_document_normal({
        "file_path": "/nonexistent/path/document.pdf",
    })
    assert result["isError"] is True
    assert result["structuredContent"]["errorType"] == "RESOURCE_NOT_FOUND"
    assert "文件不存在" in result["structuredContent"]["message"]


@patch("src.mcp_server.tools.ingest_document_normal.load_mcp_settings")
@patch("src.ingestion.pipeline.IngestionPipeline")
@patch("src.libs.loader.pdf_loader.PdfLoader")
@patch("src.mcp_server.tools.ingest_document_normal.FileIntegrityChecker")
def test_execute_success(
    mock_integrity_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_pipeline_cls: MagicMock,
    mock_load_settings: MagicMock,
    tmp_path: Path,
) -> None:
    """成功流程：PdfLoader.load → process_document → 返回 chunk_count"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    mock_settings = MagicMock()
    mock_settings.vector_store.collection_name = "report"
    mock_load_settings.return_value = mock_settings

    mock_doc = Document(id="doc_123", text="# Test", metadata={})
    mock_loader = MagicMock()
    mock_loader.load.return_value = mock_doc
    mock_loader_cls.return_value = mock_loader

    mock_pipeline = MagicMock()
    mock_pipeline.process_document.return_value = 42
    mock_pipeline_cls.return_value = mock_pipeline
    mock_integrity = MagicMock()
    mock_integrity.compute_sha256.return_value = "hash_123"
    mock_integrity.should_skip.return_value = False
    mock_integrity_cls.return_value = mock_integrity

    result = execute_ingest_document_normal({
        "file_path": str(pdf_path),
        "collection_name": "report",
    })

    assert result["isError"] is False
    assert result["structuredContent"]["chunk_count"] == 42
    assert result["structuredContent"]["collection_name"] == "report"
    assert result["structuredContent"]["skipped"] is False
    assert "42" in result["content"][0]["text"]
    assert "解析入库成功" in result["content"][0]["text"]

    mock_loader.load.assert_called_once_with(str(pdf_path))
    mock_pipeline.process_document.assert_called_once_with(mock_doc, "report", trace=ANY)
    mock_pipeline.close.assert_called_once()
    mock_integrity.mark_success.assert_called_once_with("hash_123")


@patch("src.mcp_server.tools.ingest_document_normal.load_mcp_settings")
@patch("src.ingestion.pipeline.IngestionPipeline")
@patch("src.libs.loader.pdf_loader.PdfLoader")
@patch("src.mcp_server.tools.ingest_document_normal.FileIntegrityChecker")
def test_execute_uses_default_collection(
    mock_integrity_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_pipeline_cls: MagicMock,
    mock_load_settings: MagicMock,
    tmp_path: Path,
) -> None:
    """不传 collection_name 时使用配置默认值"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    mock_settings = MagicMock()
    mock_settings.vector_store.collection_name = "default_coll"
    mock_load_settings.return_value = mock_settings

    mock_doc = Document(id="doc_123", text="# Test", metadata={})
    mock_loader_cls.return_value.load.return_value = mock_doc
    mock_pipeline_cls.return_value.process_document.return_value = 10
    mock_integrity = MagicMock()
    mock_integrity.compute_sha256.return_value = "hash_456"
    mock_integrity.should_skip.return_value = False
    mock_integrity_cls.return_value = mock_integrity

    result = execute_ingest_document_normal({"file_path": str(pdf_path)})

    assert result["isError"] is False
    assert result["structuredContent"]["skipped"] is False
    mock_pipeline_cls.return_value.process_document.assert_called_once_with(
        mock_doc, "default_coll", trace=ANY
    )
    mock_integrity.mark_success.assert_called_once_with("hash_456")


@patch("src.mcp_server.tools.ingest_document_normal.load_mcp_settings")
@patch("src.mcp_server.tools.ingest_document_normal.FileIntegrityChecker")
def test_execute_skip_when_unchanged(
    mock_integrity_cls: MagicMock,
    mock_load_settings: MagicMock,
    tmp_path: Path,
) -> None:
    """文件未变更时直接跳过，不触发入库。"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    mock_settings = MagicMock()
    mock_settings.vector_store.collection_name = "report"
    mock_load_settings.return_value = mock_settings

    mock_integrity = MagicMock()
    mock_integrity.compute_sha256.return_value = "hash_skip"
    mock_integrity.should_skip.return_value = True
    mock_integrity_cls.return_value = mock_integrity

    result = execute_ingest_document_normal({"file_path": str(pdf_path)})

    assert result["isError"] is False
    assert result["structuredContent"]["skipped"] is True
    assert result["structuredContent"]["chunk_count"] == 0
    mock_integrity.mark_success.assert_not_called()
