"""
Splitter 抽象接口模块

定义统一的文本切分接口，所有 Splitter 实现（Recursive、Semantic、Fixed 等）
都必须遵循此接口。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any


class BaseSplitter(ABC):
    """
    Splitter 抽象基类
    
    定义所有文本切分实现必须遵循的统一接口。
    无论底层使用递归切分、语义切分还是固定长度切分，
    上层代码都可以通过统一的接口调用。
    """
    
    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None
    ) -> List[str]:
        """
        将文本切分为多个片段（chunks）
        
        Args:
            text: 输入文本，需要被切分的完整文本
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
                   TraceContext 将在 F1 阶段实现，此处预留接口
            
        Returns:
            List[str]: 切分后的文本片段列表
                      - 每个元素是一个文本片段（chunk）
                      - 片段之间可能有重叠（根据配置）
                      - 片段顺序与原文顺序一致
            
        Raises:
            ValueError: 当输入文本为空或无效时
            RuntimeError: 当切分过程失败时
        """
        pass
    
    @abstractmethod
    def get_strategy(self) -> str:
        """
        获取当前使用的切分策略名称
        
        Returns:
            str: 策略名称，例如 "recursive", "semantic", "fixed"
        """
        pass
    
    @abstractmethod
    def get_chunk_size(self) -> int:
        """
        获取当前配置的块大小
        
        Returns:
            int: 块大小（字符数或 token 数）
        """
        pass
    
    @abstractmethod
    def get_chunk_overlap(self) -> int:
        """
        获取当前配置的块重叠大小
        
        Returns:
            int: 重叠大小（字符数或 token 数）
        """
        pass
