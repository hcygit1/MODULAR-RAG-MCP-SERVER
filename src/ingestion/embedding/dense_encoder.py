"""
Dense Encoder 实现

将 Chunks 批量转换为稠密向量（Dense Embeddings）：
- 提取 chunks 的文本内容
- 批量调用 BaseEmbedding 生成向量
- 确保输出向量数量与 chunks 数量一致，维度一致
"""
from typing import List, Optional, Any

from src.ingestion.models import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


class DenseEncoder:
    """
    Dense Encoder 实现
    
    将 Chunk 列表批量转换为稠密向量，用于语义检索。
    """
    
    def __init__(
        self,
        embedding: BaseEmbedding,
        batch_size: Optional[int] = None
    ):
        """
        初始化 DenseEncoder
        
        Args:
            embedding: BaseEmbedding 实例，用于生成向量
            batch_size: 批处理大小（可选），如果为 None 则一次性处理所有 chunks
        """
        if embedding is None:
            raise ValueError("embedding 不能为 None")
        
        self._embedding = embedding
        self._batch_size = batch_size
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None
    ) -> List[List[float]]:
        """
        批量将 Chunks 转换为稠密向量
        
        Args:
            chunks: Chunk 对象列表
            trace: 追踪上下文（可选）
        
        Returns:
            List[List[float]]: 向量列表，每个 Chunk 对应一个向量
                              - 长度与 chunks 相同
                              - 每个向量是一个浮点数列表
                              - 所有向量维度一致
        
        Raises:
            ValueError: 当 chunks 为空或包含无效 Chunk 时
            RuntimeError: 当 Embedding 调用失败时
        """
        if not chunks:
            raise ValueError("chunks 列表不能为空")
        
        # 提取文本列表
        texts = [chunk.text for chunk in chunks]
        
        # 批量生成向量
        if self._batch_size is None or self._batch_size <= 0:
            # 一次性处理所有文本
            vectors = self._embedding.embed(texts, trace=trace)
        else:
            # 分批处理
            vectors = []
            for i in range(0, len(texts), self._batch_size):
                batch_texts = texts[i:i + self._batch_size]
                batch_vectors = self._embedding.embed(batch_texts, trace=trace)
                vectors.extend(batch_vectors)
        
        # 验证输出数量
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"向量数量 ({len(vectors)}) 与 chunks 数量 ({len(chunks)}) 不一致"
            )
        
        # 验证维度一致性
        if vectors:
            expected_dim = len(vectors[0])
            for idx, vector in enumerate(vectors):
                if len(vector) != expected_dim:
                    raise RuntimeError(
                        f"向量维度不一致: chunk[{idx}] 的向量维度为 {len(vector)}, "
                        f"期望维度为 {expected_dim}"
                    )
        
        return vectors
    
    def get_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            int: 向量维度
        """
        return self._embedding.get_dimension()
    
    def get_model_name(self) -> str:
        """
        获取使用的模型名称
        
        Returns:
            str: 模型名称
        """
        return self._embedding.get_model_name()
    
    def get_provider(self) -> str:
        """
        获取使用的 provider 名称
        
        Returns:
            str: provider 名称
        """
        return self._embedding.get_provider()
