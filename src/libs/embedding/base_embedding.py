"""
Embedding 抽象接口模块

定义统一的 Embedding 接口，所有 Embedding 实现（OpenAI、Local、Ollama 等）
都必须遵循此接口。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any


class BaseEmbedding(ABC):
    """
    Embedding 抽象基类
    
    定义所有 Embedding 实现必须遵循的统一接口。
    无论底层使用 OpenAI API、本地模型还是 Ollama，
    上层代码都可以通过统一的接口调用。
    """
    
    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None
    ) -> List[List[float]]:
        """
        批量将文本转换为向量
        
        Args:
            texts: 文本列表，每个元素是一个字符串
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
                   TraceContext 将在 F1 阶段实现，此处预留接口
            
        Returns:
            List[List[float]]: 向量列表，每个文本对应一个向量
                              - 长度与 texts 相同
                              - 每个向量是一个浮点数列表
                              - 向量维度由具体实现决定
            
        Raises:
            ValueError: 当文本列表为空或包含无效文本时
            RuntimeError: 当 Embedding 调用失败时（网络错误、API 错误等）
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        获取当前使用的模型名称
        
        Returns:
            str: 模型名称，例如 "text-embedding-3-small", "bge-large-zh-v1.5"
        """
        pass
    
    @abstractmethod
    def get_provider(self) -> str:
        """
        获取当前使用的 provider 名称
        
        Returns:
            str: provider 名称，例如 "openai", "local", "ollama"
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            int: 向量维度，例如 1536, 1024
        """
        pass
