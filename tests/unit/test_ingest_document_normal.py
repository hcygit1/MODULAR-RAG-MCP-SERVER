"""ingest_document_normal 单元测试"""
from pathlib import Path
from unittest.mock import MagicMock, patch

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
def test_execute_success(
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

    result = execute_ingest_document_normal({
        "file_path": str(pdf_path),
        "collection_name": "report",
    })

    assert result["isError"] is False
    assert result["structuredContent"]["chunk_count"] == 42
    assert result["structuredContent"]["collection_name"] == "report"
    assert "42" in result["content"][0]["text"]
    assert "普通解析" in result["content"][0]["text"]

    mock_loader.load.assert_called_once_with(str(pdf_path))
    mock_pipeline.process_document.assert_called_once_with(mock_doc, "report")
    mock_pipeline.close.assert_called_once()


@patch("src.mcp_server.tools.ingest_document_normal.load_mcp_settings")
@patch("src.ingestion.pipeline.IngestionPipeline")
@patch("src.libs.loader.pdf_loader.PdfLoader")
def test_execute_uses_default_collection(
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

    result = execute_ingest_document_normal({"file_path": str(pdf_path)})

    assert result["isError"] is False
    mock_pipeline_cls.return_value.process_document.assert_called_once_with(
        mock_doc, "default_coll"
    )
