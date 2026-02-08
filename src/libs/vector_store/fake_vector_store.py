"""
Fake VectorStore 实现（用于测试）

提供一个简单的 Fake VectorStore 实现，用于测试工厂路由逻辑和契约测试，
不进行真实的数据库操作。
"""
from typing import List, Dict, Any, Optional
import math

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult
)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        float: 余弦相似度，范围 [-1, 1]
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"向量维度不匹配: {len(vec1)} vs {len(vec2)}")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


class FakeVectorStore(BaseVectorStore):
    """
    Fake VectorStore 实现
    
    用于测试和开发阶段，使用内存存储向量记录。
    提供基本的 upsert 和 query 功能，用于验证接口契约。
    """
    
    def __init__(
        self,
        backend: str = "fake",
        collection_name: str = "test_collection"
    ):
        """
        初始化 Fake VectorStore
        
        Args:
            backend: 后端名称
            collection_name: 集合名称
        """
        self._backend = backend
        self._collection_name = collection_name
        self._records: Dict[str, VectorRecord] = {}  # id -> record
    
    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None
    ) -> None:
        """
        批量插入或更新向量记录（内存存储）
        
        Args:
            records: 向量记录列表
            trace: 追踪上下文（可选，Fake 实现中不使用）
            
        Raises:
            ValueError: 当记录格式不正确时
        """
        if not records:
            raise ValueError("记录列表不能为空")
        
        for record in records:
            if not isinstance(record, VectorRecord):
                raise ValueError(f"记录必须是 VectorRecord 类型，得到: {type(record)}")
            
            if not record.id:
                raise ValueError("记录 ID 不能为空")
            
            if not record.vector:
                raise ValueError("记录向量不能为空")
            
            if not record.text:
                raise ValueError("记录文本不能为空")
            
            # 幂等操作：相同 ID 的记录会被覆盖
            self._records[record.id] = record
    
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None
    ) -> List[QueryResult]:
        """
        向量相似度查询（使用余弦相似度）
        
        Args:
            vector: 查询向量
            top_k: 返回最相似的 top_k 条记录
            filters: 元数据过滤条件（可选）
            trace: 追踪上下文（可选，Fake 实现中不使用）
            
        Returns:
            List[QueryResult]: 查询结果列表，按相似度分数降序排列
            
        Raises:
            ValueError: 当向量为空或 top_k <= 0 时
        """
        if not vector:
            raise ValueError("查询向量不能为空")
        
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")
        
        # 计算所有记录的相似度
        candidates = []
        
        for record in self._records.values():
            # 应用元数据过滤
            if filters:
                if not self._matches_filters(record.metadata, filters):
                    continue
            
            # 计算相似度
            score = cosine_similarity(vector, record.vector)
            candidates.append(QueryResult(
                id=record.id,
                score=score,
                text=record.text,
                metadata=record.metadata
            ))
        
        # 按分数降序排序，返回 top_k
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        检查元数据是否匹配过滤条件
        
        Args:
            metadata: 记录的元数据
            filters: 过滤条件
            
        Returns:
            bool: 是否匹配
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def get_backend(self) -> str:
        """获取后端名称"""
        return self._backend
    
    def get_collection_name(self) -> str:
        """获取集合名称"""
        return self._collection_name
