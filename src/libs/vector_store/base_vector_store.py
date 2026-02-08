"""
VectorStore 抽象接口模块

定义统一的向量存储接口，所有 VectorStore 实现（Chroma、Qdrant、Pinecone 等）
都必须遵循此接口。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorRecord:
    """
    向量记录数据结构
    
    表示一条完整的向量存储记录，包含向量、文本内容和元数据。
    """
    
    def __init__(
        self,
        id: str,
        vector: List[float],
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化向量记录
        
        Args:
            id: 记录唯一标识符
            vector: 向量（dense vector）
            text: 文本内容
            metadata: 元数据字典（可选）
        """
        self.id = id
        self.vector = vector
        self.text = text
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "vector": self.vector,
            "text": self.text,
            "metadata": self.metadata
        }


class QueryResult:
    """
    查询结果数据结构
    
    表示一次向量查询的返回结果。
    """
    
    def __init__(
        self,
        id: str,
        score: float,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化查询结果
        
        Args:
            id: 记录唯一标识符
            score: 相似度分数
            text: 文本内容
            metadata: 元数据字典（可选）
        """
        self.id = id
        self.score = score
        self.text = text
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata
        }


class BaseVectorStore(ABC):
    """
    VectorStore 抽象基类
    
    定义所有向量存储实现必须遵循的统一接口。
    无论底层使用 Chroma、Qdrant、Pinecone 还是其他向量数据库，
    上层代码都可以通过统一的接口调用。
    """
    
    @abstractmethod
    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None
    ) -> None:
        """
        批量插入或更新向量记录（幂等操作）
        
        Args:
            records: 向量记录列表，每个记录包含 id、vector、text、metadata
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
                   TraceContext 将在 F1 阶段实现，此处预留接口
        
        Raises:
            ValueError: 当记录格式不正确时
            RuntimeError: 当存储操作失败时（数据库错误、网络错误等）
        """
        pass
    
    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None
    ) -> List[QueryResult]:
        """
        向量相似度查询
        
        Args:
            vector: 查询向量
            top_k: 返回最相似的 top_k 条记录
            filters: 元数据过滤条件（可选），例如 {"source": "doc1.pdf"}
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
        
        Returns:
            List[QueryResult]: 查询结果列表，按相似度分数降序排列
                              - 每个结果包含 id、score、text、metadata
                              - 结果数量 <= top_k
        
        Raises:
            ValueError: 当向量维度不匹配或 top_k <= 0 时
            RuntimeError: 当查询操作失败时
        """
        pass
    
    @abstractmethod
    def get_backend(self) -> str:
        """
        获取当前使用的后端名称
        
        Returns:
            str: 后端名称，例如 "chroma", "qdrant", "pinecone"
        """
        pass
    
    @abstractmethod
    def get_collection_name(self) -> str:
        """
        获取当前集合名称
        
        Returns:
            str: 集合名称
        """
        pass
