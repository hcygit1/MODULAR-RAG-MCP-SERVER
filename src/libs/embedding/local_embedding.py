"""
Local Embedding 实现（占位/适配层）

本地 Embedding 模型的占位实现。
当前使用基于哈希的向量生成（类似 FakeEmbedding），保证链路可跑。
后续可以接入真实的本地模型（如 BGE、Ollama Embedding 等）。
"""
import hashlib
from typing import List, Optional, Any

from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.settings import EmbeddingConfig


class LocalEmbedding(BaseEmbedding):
    """
    Local Embedding 实现（占位/适配层）
    
    当前使用基于哈希的向量生成，保证链路可跑。
    后续可以接入真实的本地模型（如 BGE、Ollama Embedding 等）。
    
    支持的模型维度映射（常见本地模型）：
    - bge-large-zh-v1.5: 1024
    - bge-base-zh-v1.5: 768
    - bge-small-zh-v1.5: 512
    - 默认: 768
    """
    
    # 常见本地模型的维度映射
    MODEL_DIMENSIONS = {
        "bge-large-zh-v1.5": 1024,
        "bge-base-zh-v1.5": 768,
        "bge-small-zh-v1.5": 512,
        "bge-large-en-v1.5": 1024,
        "bge-base-en-v1.5": 768,
        "bge-small-en-v1.5": 512,
    }
    
    def __init__(self, config: EmbeddingConfig):
        """
        初始化 Local Embedding
        
        Args:
            config: Embedding 配置对象
        """
        if not config.model:
            raise ValueError("Local Embedding model 名称不能为空")
        
        self._config = config
        self._provider = config.provider.lower()  # 从配置中获取 provider
        self._model = config.model
        
        # 获取模型维度（从映射表或使用默认值）
        self._dimension = self.MODEL_DIMENSIONS.get(
            self._model.lower(),
            768  # 默认维度（bge-base 系列）
        )
        
        # 如果配置了 local_model_path，可以在这里加载模型
        # 当前占位实现暂不使用
        self._model_path = config.local_model_path
        self._device = config.device
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None
    ) -> List[List[float]]:
        """
        批量将文本转换为向量
        
        当前实现：使用基于哈希的向量生成（占位实现）
        后续：可以接入真实的本地模型（如 BGE、Ollama Embedding 等）
        
        Args:
            texts: 文本列表，每个元素是一个字符串
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
        
        Returns:
            List[List[float]]: 向量列表，每个文本对应一个向量
        
        Raises:
            ValueError: 当文本列表为空或包含无效文本时
            RuntimeError: 当 Embedding 调用失败时（模型加载错误等）
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        # 验证文本格式
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"文本 {i} 必须是字符串类型，得到: {type(text)}")
            if not text.strip():
                raise ValueError(f"文本 {i} 不能为空")
        
        try:
            # 当前占位实现：使用基于哈希的向量生成
            # 后续可以替换为真实的本地模型调用
            return self._generate_vectors_from_texts(texts)
        except Exception as e:
            raise RuntimeError(
                f"Local Embedding 调用失败 (provider={self._provider}, model={self._model}): {str(e)}"
            ) from e
    
    def _generate_vectors_from_texts(self, texts: List[str]) -> List[List[float]]:
        """
        基于文本内容生成稳定的向量（占位实现）
        
        使用 SHA256 哈希将文本映射到固定维度的向量。
        相同文本总是返回相同的向量。
        
        后续可以替换为真实的本地模型调用，例如：
        - BGE 模型：使用 sentence-transformers 库
        - Ollama Embedding：调用本地 Ollama API
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        vectors = []
        for text in texts:
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
