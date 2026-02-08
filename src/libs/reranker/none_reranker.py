"""
None Reranker 实现

提供一个不改变排序的 Reranker 实现，作为默认回退选项。
当配置 backend=none 时使用此实现。
"""
from typing import List, Optional, Any

from src.libs.vector_store.base_vector_store import QueryResult
from src.libs.reranker.base_reranker import BaseReranker


class NoneReranker(BaseReranker):
    """
    None Reranker 实现
    
    不进行任何重排序，直接返回原始候选列表。
    作为默认回退选项，当不需要重排序时使用。
    """
    
    def __init__(self):
        """初始化 None Reranker"""
        self._backend = "none"
    
    def rerank(
        self,
        query: str,
        candidates: List[QueryResult],
        trace: Optional[Any] = None
    ) -> List[QueryResult]:
        """
        不改变排序，直接返回原始候选列表
        
        Args:
            query: 查询文本（此实现中不使用）
            candidates: 候选结果列表
            trace: 追踪上下文（可选，此实现中不使用）
        
        Returns:
            List[QueryResult]: 与输入相同的候选列表（不改变顺序）
        
        Raises:
            ValueError: 当候选列表为空时
        """
        if not candidates:
            raise ValueError("候选列表不能为空")
        
        # 不改变排序，返回列表的副本（深拷贝 QueryResult 对象）
        # 确保修改返回结果不会影响原始候选列表
        return [
            QueryResult(
                id=candidate.id,
                score=candidate.score,
                text=candidate.text,
                metadata=candidate.metadata.copy() if candidate.metadata else {}
            )
            for candidate in candidates
        ]
    
    def get_backend(self) -> str:
        """获取 backend 名称"""
        return self._backend
