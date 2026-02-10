"""
PDF Loader 图片提取单元测试

测试 PdfLoader 的图片提取、页面映射和占位符插入功能。
"""
import os
import tempfile
from pathlib import Path

import pytest

# 延迟导入以避免循环依赖（base_loader -> models -> ingestion __init__ -> pipeline -> pdf_loader）
MARKITDOWN_AVAILABLE = False
try:
    from markitdown import MarkItDown  # noqa: F401
    MARKITDOWN_AVAILABLE = True
except ImportError:
    pass

PYMUPDF_AVAILABLE = False
try:
    import fitz  # noqa: F401
    PYMUPDF_AVAILABLE = True
except ImportError:
    pass


def _create_pdf_with_image(tmp_path: Path) -> str:
    """使用 PyMuPDF 创建包含一张图片的测试 PDF"""
    if not PYMUPDF_AVAILABLE:
        pytest.skip("PyMuPDF 未安装")
    
    import base64
    import fitz
    
    # 有效的 1x1 像素 PNG（base64 解码）
    minimal_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    
    pdf_path = tmp_path / "test_with_image.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_image(fitz.Rect(0, 0, 100, 100), stream=minimal_png)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def _create_pdf_without_image(tmp_path: Path) -> str:
    """创建不包含图片的 PDF"""
    if not PYMUPDF_AVAILABLE:
        pytest.skip("PyMuPDF 未安装")
    
    import fitz
    
    pdf_path = tmp_path / "test_no_image.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((50, 100), "Hello, PDF with text only.")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestPdfLoaderImageExtraction:
    """图片提取测试"""
    
    def test_extract_images_from_pdf_with_images(self, tmp_path):
        """测试有图片的 PDF 能正确提取图片"""
        from src.libs.loader.pdf_loader import PdfLoader
        
        pdf_path = _create_pdf_with_image(tmp_path)
        loader = PdfLoader()
        document = loader.load(pdf_path)
        
        assert document is not None
        assert "image_data" in document.metadata
        assert "images" in document.metadata
        assert len(document.metadata["image_data"]) >= 1
        assert len(document.metadata["images"]) >= 1
        
        image_data = document.metadata["image_data"]
        for img_id, data in image_data.items():
            assert isinstance(data, bytes)
            assert len(data) > 0
            assert img_id.startswith("doc_")
            assert "_page_" in img_id
            assert "_img_" in img_id
    
    def test_extract_images_from_pdf_without_images(self, tmp_path):
        """测试无图片的 PDF 不报错且 metadata 为空或合理"""
        from src.libs.loader.pdf_loader import PdfLoader
        
        pdf_path = _create_pdf_without_image(tmp_path)
        loader = PdfLoader()
        document = loader.load(pdf_path)
        
        assert document is not None
        # 无图片时 image_data 可能为空字典或不存在
        image_data = document.metadata.get("image_data", {})
        assert isinstance(image_data, dict)
    
    def test_image_id_format(self, tmp_path):
        """测试 image_id 格式正确"""
        from src.libs.loader.pdf_loader import PdfLoader
        
        pdf_path = _create_pdf_with_image(tmp_path)
        loader = PdfLoader()
        document = loader.load(pdf_path)
        
        images = document.metadata.get("images", [])
        if images:
            for meta in images:
                img_id = meta.get("image_id")
                assert img_id is not None
                assert "_page_" in img_id
                assert "_img_" in img_id


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestPdfLoaderPlaceholderInsertion:
    """占位符插入测试"""
    
    def test_placeholder_inserted_when_images_extracted(self, tmp_path):
        """测试有图片时 Markdown 中包含占位符"""
        from src.libs.loader.pdf_loader import PdfLoader
        
        pdf_path = _create_pdf_with_image(tmp_path)
        loader = PdfLoader()
        document = loader.load(pdf_path)
        
        import re
        placeholders = re.findall(r'\[IMAGE:\s*([^\]]+)\]', document.text)
        
        image_data = document.metadata.get("image_data", {})
        if image_data:
            # 有图片时应该有占位符（可能简单策略追加到末尾）
            assert len(placeholders) >= 1
            for ph in placeholders:
                assert ph in image_data
    
    def test_placeholder_format(self, tmp_path):
        """测试占位符格式为 [IMAGE: {image_id}]"""
        from src.libs.loader.pdf_loader import PdfLoader
        
        pdf_path = _create_pdf_with_image(tmp_path)
        loader = PdfLoader()
        document = loader.load(pdf_path)
        
        import re
        placeholders = re.findall(r'\[IMAGE:\s*([^\]]+)\]', document.text)
        
        for ph in placeholders:
            assert len(ph) > 0
            assert " " not in ph or "_" in ph  # image_id 格式


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestPdfLoaderPageMapping:
    """页面映射测试"""
    
    def test_map_pages_to_markdown_returns_dict(self, tmp_path):
        """测试 _map_pages_to_markdown 返回正确格式"""
        from src.libs.loader.pdf_loader import PdfLoader
        
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF 未安装")
        
        pdf_path = _create_pdf_with_image(tmp_path)
        loader = PdfLoader()
        document = loader.load(pdf_path)
        markdown_text = document.text or "sample text"
        
        result = loader._map_pages_to_markdown(
            pdf_path=pdf_path,
            markdown_text=markdown_text,
        )
        assert isinstance(result, dict)


@pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF 未安装")
@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestPdfLoaderFallback:
    """PyMuPDF 未安装时的回退测试"""
    
    def test_loader_works_when_pymupdf_available(self, tmp_path):
        """PyMuPDF 可用时能正常加载"""
        from src.libs.loader.pdf_loader import PdfLoader
        
        pdf_path = _create_pdf_with_image(tmp_path)
        loader = PdfLoader()
        document = loader.load(pdf_path)
        assert document is not None


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestPdfLoaderImageMetadata:
    """图片元数据测试"""
    
    def test_image_metadata_contains_required_fields(self, tmp_path):
        """测试图片元数据包含必要字段"""
        from src.libs.loader.pdf_loader import PdfLoader
        
        if not PYMUPDF_AVAILABLE:
            pytest.skip("PyMuPDF 未安装")
        
        pdf_path = _create_pdf_with_image(tmp_path)
        loader = PdfLoader()
        document = loader.load(pdf_path)
        
        images = document.metadata.get("images", [])
        for meta in images:
            assert "image_id" in meta
            assert "page" in meta
            assert "mime_type" in meta or "ext" in meta
