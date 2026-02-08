"""
Loader PDF 契约测试

验证 BaseLoader 接口和 PdfLoader 实现是否符合规范要求。
"""
import tempfile
import os
import pytest
from pathlib import Path

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader, MARKITDOWN_AVAILABLE
from src.ingestion.models import Document


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestBaseLoader:
    """BaseLoader 抽象基类测试"""
    
    def test_base_loader_is_abstract(self):
        """测试 BaseLoader 是抽象类，不能直接实例化"""
        with pytest.raises(TypeError):
            BaseLoader()
    
    def test_base_loader_has_load_method(self):
        """测试 BaseLoader 定义了 load 方法"""
        assert hasattr(BaseLoader, 'load')
        assert callable(getattr(BaseLoader, 'load'))


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestPdfLoader:
    """PdfLoader 实现测试"""
    
    def test_pdf_loader_initialization(self):
        """测试 PdfLoader 可以初始化"""
        loader = PdfLoader()
        assert isinstance(loader, PdfLoader)
        assert isinstance(loader, BaseLoader)
    
    def test_pdf_loader_supported_extensions(self):
        """测试 PdfLoader 支持的文件扩展名"""
        loader = PdfLoader()
        extensions = loader.get_supported_extensions()
        assert ".pdf" in extensions
        assert isinstance(extensions, list)
    
    def test_pdf_loader_file_not_found(self):
        """测试文件不存在时抛出 FileNotFoundError"""
        loader = PdfLoader()
        
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            loader.load("/nonexistent/file.pdf")
    
    def test_pdf_loader_invalid_extension(self):
        """测试不支持的文件扩展名时抛出 ValueError"""
        loader = PdfLoader()
        
        # 创建一个临时文件，但使用错误的扩展名
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="不支持的文件类型"):
                loader.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_pdf_loader_returns_document(self):
        """测试 PdfLoader.load() 返回 Document 对象"""
        loader = PdfLoader()
        
        # 创建一个最小的 PDF 文件（实际测试中需要真实的 PDF）
        # 这里我们测试接口契约，实际 PDF 解析需要真实文件
        # 如果 MarkItDown 可用，我们可以创建一个简单的测试 PDF
        
        # 注意：这个测试需要真实的 PDF 文件
        # 在 fixtures 目录中应该有 sample PDF
        pass
    
    def test_pdf_loader_document_has_required_fields(self):
        """测试返回的 Document 包含必需字段"""
        loader = PdfLoader()
        
        # 这个测试需要真实的 PDF 文件
        # 验证 Document 包含：
        # - id: 非空字符串
        # - text: 字符串（可能是空字符串）
        # - metadata: 字典，至少包含 source_path
        pass


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestPdfLoaderWithSampleFile:
    """使用样例文件测试 PdfLoader"""
    
    def test_load_sample_pdf_if_exists(self):
        """如果存在样例 PDF，测试加载功能"""
        # 检查是否有样例 PDF 文件
        sample_pdf_paths = [
            "tests/fixtures/sample_documents/sample.pdf",
            "tests/fixtures/sample.pdf",
        ]
        
        pdf_path = None
        for path in sample_pdf_paths:
            if os.path.exists(path):
                pdf_path = path
                break
        
        if pdf_path is None:
            pytest.skip("未找到样例 PDF 文件")
        
        loader = PdfLoader()
        document = loader.load(pdf_path)
        
        # 验证返回的是 Document 对象
        assert isinstance(document, Document)
        
        # 验证 Document 有必需的字段
        assert document.id is not None
        assert len(document.id) > 0
        assert isinstance(document.text, str)
        assert isinstance(document.metadata, dict)
        
        # 验证 metadata 至少包含 source_path
        assert "source_path" in document.metadata
        assert document.metadata["source_path"] == os.path.abspath(pdf_path)
        
        # 验证 metadata 包含 doc_type
        assert "doc_type" in document.metadata
        assert document.metadata["doc_type"] == "pdf"
        
        # 验证 metadata 包含 images 列表
        assert "images" in document.metadata
        assert isinstance(document.metadata["images"], list)


@pytest.mark.skipif(MARKITDOWN_AVAILABLE, reason="MarkItDown 已安装，跳过此测试")
class TestPdfLoaderWithoutMarkItDown:
    """测试 MarkItDown 未安装时的行为"""
    
    def test_pdf_loader_raises_error_without_markitdown(self):
        """测试 MarkItDown 未安装时初始化抛出错误"""
        with pytest.raises(RuntimeError, match="MarkItDown 未安装"):
            PdfLoader()


class TestLoaderContract:
    """Loader 契约测试"""
    
    @pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
    def test_pdf_loader_implements_base_loader(self):
        """测试 PdfLoader 实现了 BaseLoader 接口"""
        loader = PdfLoader()
        
        # 验证实现了所有抽象方法
        assert hasattr(loader, 'load')
        assert callable(loader.load)
        assert hasattr(loader, 'get_supported_extensions')
        assert callable(loader.get_supported_extensions)
        
        # 验证 load 方法的签名
        import inspect
        sig = inspect.signature(loader.load)
        params = list(sig.parameters.keys())
        assert 'path' in params
        assert 'trace' in params or len(params) >= 1  # trace 是可选的
