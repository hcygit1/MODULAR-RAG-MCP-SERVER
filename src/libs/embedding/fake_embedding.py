"""
Fake Embedding 实现（用于测试）

提供一个简单的 Fake Embedding 实现，用于测试工厂路由逻辑，
不进行真实的 Embedding API 调用。返回稳定、可预测的向量。
"""
import hashlib
from typing import List, Optional, Any

from src.libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    """
    Fake Embedding 实现
    
    用于测试和开发阶段，返回稳定、可预测的向量，不进行真实的 API 调用。
    相同文本总是返回相同的向量（基于文本哈希生成）。
    """
    
    def __init__(
        self,
        provider: str = "fake",
        model: str = "fake-model",
        dimension: int = 128
    ):
        """
        初始化 Fake Embedding
        
        Args:
            provider: provider 名称
            model: 模型名称
            dimension: 向量维度
        """
        self._provider = provider
        self._model = model
        self._dimension = dimension
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None
    ) -> List[List[float]]:
        """
        返回基于文本哈希生成的稳定向量
        
        Args:
            texts: 文本列表
            trace: 追踪上下文（可选，Fake 实现中不使用）
            
        Returns:
            List[List[float]]: 向量列表，每个文本对应一个向量
            
        Raises:
            ValueError: 当文本列表为空时
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        vectors = []
        for text in texts:
            # 使用文本哈希生成稳定的向量
            # 相同文本总是返回相同的向量
            vector = self._generate_vector_from_text(text)
            vectors.append(vector)
        
        return vectors
    
    def _generate_vector_from_text(self, text: str) -> List[float]:
        """
        基于文本内容生成稳定的向量
        
        使用 SHA256 哈希将文本映射到固定维度的向量。
        相同文本总是返回相同的向量。
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量（浮点数列表）
        """
        # 使用 SHA256 生成文本哈希
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # 将哈希字节转换为向量
        # 使用多个字节组合确保向量维度足够
        vector = []
        for i in range(self._dimension):
            # 循环使用哈希字节，确保维度匹配
            byte_index = i % len(hash_bytes)
            # 将字节值归一化到 [-1, 1] 范围
            value = (hash_bytes[byte_index] / 255.0) * 2.0 - 1.0
            vector.append(value)
        
        return vector
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self._model
    
    def get_provider(self) -> str:
        """获取 provider 名称"""
        return self._provider
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self._dimension
