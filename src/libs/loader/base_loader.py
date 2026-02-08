"""
Loader 抽象接口模块

定义统一的文档加载接口，所有 Loader 实现（PDF、Markdown、Code 等）
都必须遵循此接口。
"""
from abc import ABC, abstractmethod
from typing import Optional, Any

from src.ingestion.models import Document


class BaseLoader(ABC):
    """
    Loader 抽象基类
    
    定义所有文档加载实现必须遵循的统一接口。
    无论底层加载 PDF、Markdown 还是代码文件，
    上层代码都可以通过统一的接口调用。
    """
    
    @abstractmethod
    def load(self, path: str, trace: Optional[Any] = None) -> Document:
        """
        加载文档并转换为统一的 Document 对象
        
        Args:
            path: 文件路径，需要加载的文档路径
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
                   TraceContext 将在 F1 阶段实现，此处预留接口
        
        Returns:
            Document: 文档对象，包含：
                     - id: 文档唯一标识符
                     - text: 文档文本内容（通常是 Markdown 格式）
                     - metadata: 元数据字典，至少包含：
                       * source_path: 源文件路径
                       * doc_type: 文档类型（如 "pdf", "markdown"）
                       * 其他可选字段：title, page, images 等
        
        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件路径无效时
            RuntimeError: 当文档加载或解析失败时
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """
        获取支持的文件扩展名列表
        
        Returns:
            list[str]: 支持的文件扩展名列表，例如 [".pdf", ".pdfx"]
        """
        pass
