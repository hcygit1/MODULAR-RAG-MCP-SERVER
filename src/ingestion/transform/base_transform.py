"""
Transform 抽象接口模块

定义统一的 Transform 接口，所有 Transform 实现（ChunkRefiner、MetadataEnricher、ImageCaptioner 等）
都必须遵循此接口。
"""
from abc import ABC, abstractmethod
from typing import Optional, Any

from src.ingestion.models import Chunk


class BaseTransform(ABC):
    """
    Transform 抽象基类
    
    定义所有 Transform 实现必须遵循的统一接口。
    无论底层是规则处理、LLM 增强还是多模态处理，
    上层代码都可以通过统一的接口调用。
    """
    
    @abstractmethod
    def transform(
        self,
        chunk: Chunk,
        trace: Optional[Any] = None
    ) -> Chunk:
        """
        对 Chunk 进行转换/增强处理
        
        Args:
            chunk: 输入的 Chunk 对象
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
                   TraceContext 将在 F1 阶段实现，此处预留接口
        
        Returns:
            Chunk: 处理后的 Chunk 对象
                  - 可以修改 text、metadata 等字段
                  - 应该保留原始的 id、start_offset、end_offset 等定位信息
                  - 可以在 metadata 中记录处理信息（如降级原因）
        
        Raises:
            ValueError: 当 Chunk 无效时
            RuntimeError: 当处理过程失败时（但应该尽量降级而不是抛出异常）
        """
        pass
    
    @abstractmethod
    def get_transform_name(self) -> str:
        """
        获取当前 Transform 的名称
        
        Returns:
            str: Transform 名称，例如 "chunk_refiner", "metadata_enricher"
        """
        pass
