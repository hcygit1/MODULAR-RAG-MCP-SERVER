"""
PDF Loader 实现

使用 MarkItDown 将 PDF 文件转换为 Markdown 格式的 Document 对象。
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Any, Dict

from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader

# 尝试导入 MarkItDown
try:
    from markitdown import MarkItDown  # type: ignore
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    MarkItDown = None


class PdfLoader(BaseLoader):
    """
    PDF Loader 实现
    
    使用 MarkItDown 将 PDF 文件转换为 Markdown 格式。
    支持提取文本、图片引用和基础元数据。
    """
    
    def __init__(self):
        """初始化 PDF Loader"""
        if not MARKITDOWN_AVAILABLE:
            raise RuntimeError(
                "MarkItDown 未安装。请安装: pip install markitdown"
            )
        self._md = MarkItDown()
        self._supported_extensions = [".pdf"]
    
    def load(self, path: str, trace: Optional[Any] = None) -> Document:
        """
        加载 PDF 文件并转换为 Document 对象
        
        Args:
            path: PDF 文件路径
            trace: 追踪上下文（可选）
        
        Returns:
            Document: 包含 Markdown 文本和元数据的文档对象
        
        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件路径无效或不是 PDF 文件时
            RuntimeError: 当 PDF 解析失败时
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        
        # 验证文件扩展名
        file_ext = Path(path).suffix.lower()
        if file_ext not in self._supported_extensions:
            raise ValueError(
                f"不支持的文件类型: {file_ext}。"
                f"支持的扩展名: {self._supported_extensions}"
            )
        
        try:
            # 使用 MarkItDown 转换 PDF 为 Markdown
            result = self._md.convert(path)
            
            # 提取文本内容
            markdown_text = result.text_content if hasattr(result, 'text_content') else str(result)
            
            # 如果没有文本内容，使用空字符串
            if not markdown_text or not markdown_text.strip():
                markdown_text = ""
            
            # 生成文档 ID（基于文件路径的哈希）
            doc_id = self._generate_doc_id(path)
            
            # 构建元数据
            metadata = self._build_metadata(path, result)
            
            return Document(
                id=doc_id,
                text=markdown_text,
                metadata=metadata
            )
            
        except Exception as e:
            raise RuntimeError(
                f"PDF 解析失败: {path}。错误: {str(e)}"
            ) from e
    
    def _generate_doc_id(self, path: str) -> str:
        """
        生成文档唯一标识符
        
        基于文件路径生成稳定的文档 ID。
        
        Args:
            path: 文件路径
        
        Returns:
            str: 文档 ID
        """
        # 使用绝对路径确保唯一性
        abs_path = os.path.abspath(path)
        # 使用 SHA256 哈希的前 16 个字符作为 ID
        hash_obj = hashlib.sha256(abs_path.encode('utf-8'))
        return f"doc_{hash_obj.hexdigest()[:16]}"
    
    def _build_metadata(
        self,
        path: str,
        result: Any
    ) -> Dict[str, Any]:
        """
        构建文档元数据
        
        Args:
            path: 文件路径
            result: MarkItDown 转换结果
        
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata: Dict[str, Any] = {
            "source_path": os.path.abspath(path),
            "doc_type": "pdf",
        }
        
        # 提取文件名（不含扩展名）作为可能的标题
        file_name = Path(path).stem
        if file_name:
            metadata["title"] = file_name
        
        # 尝试提取图片引用（如果 MarkItDown 支持）
        if hasattr(result, 'images') and result.images:
            metadata["images"] = result.images
        elif hasattr(result, 'image_refs') and result.image_refs:
            metadata["images"] = result.image_refs
        else:
            metadata["images"] = []
        
        # 尝试提取其他元数据
        if hasattr(result, 'metadata') and result.metadata:
            # 合并额外的元数据
            for key, value in result.metadata.items():
                if key not in metadata:
                    metadata[key] = value
        
        return metadata
    
    def get_supported_extensions(self) -> list[str]:
        """
        获取支持的文件扩展名列表
        
        Returns:
            list[str]: 支持的文件扩展名列表
        """
        return self._supported_extensions.copy()
