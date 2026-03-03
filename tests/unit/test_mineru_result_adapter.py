"""MinerUResultAdapter 单元测试"""
import pytest

from src.ingestion.models import Document
from src.libs.loader.mineru_cloud_client import MinerURawResult
from src.libs.loader.mineru_result_adapter import to_document


def test_to_document_basic():
    """无图片时正确转为 Document"""
    raw = MinerURawResult(
        markdown_text="# Hello\n\nSome text.",
        source_path="/tmp/test.pdf",
        images=[],
        content_list=None,
    )
    doc = to_document(raw)
    assert isinstance(doc, Document)
    assert doc.text == "# Hello\n\nSome text."
    assert doc.metadata["source_path"] == "/tmp/test.pdf"
    assert doc.metadata["doc_type"] == "pdf"
    assert doc.metadata["image_data"] == {}
    assert doc.metadata["images"] == []


def test_to_document_with_images():
    """有图片时替换占位符并构建 image_data/images"""
    raw = MinerURawResult(
        markdown_text="Text before\n\n![](images/fig1.png)\n\nText after",
        source_path="/tmp/report.pdf",
        images=[
            ("images/fig1.png", 0, b"fake_png_bytes"),
        ],
        content_list=None,
    )
    doc = to_document(raw)
    assert "[IMAGE:" in doc.text
    assert "![](images/fig1.png)" not in doc.text
    assert len(doc.metadata["image_data"]) == 1
    assert len(doc.metadata["images"]) == 1
    img_id = list(doc.metadata["image_data"].keys())[0]
    assert doc.metadata["image_data"][img_id] == b"fake_png_bytes"
    assert doc.metadata["images"][0]["image_id"] == img_id
    assert doc.metadata["images"][0]["page"] == 0


def test_to_document_with_custom_doc_id():
    """可传入自定义 doc_id"""
    raw = MinerURawResult(
        markdown_text="x",
        source_path="/tmp/a.pdf",
        images=[],
    )
    doc = to_document(raw, doc_id="custom_123")
    assert doc.id == "custom_123"


def test_to_document_image_ref_with_alt():
    """![alt](path) 格式也能替换"""
    raw = MinerURawResult(
        markdown_text="![Figure 1](images/fig1.jpg)",
        source_path="/tmp/a.pdf",
        images=[
            ("images/fig1.jpg", 1, b"jpeg_bytes"),
        ],
    )
    doc = to_document(raw)
    assert "[IMAGE:" in doc.text
    assert len(doc.metadata["image_data"]) == 1
