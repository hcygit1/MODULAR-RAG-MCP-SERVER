"""
Reranker 抽象接口模块

定义统一的重排序接口，所有 Reranker 实现（None、CrossEncoder、LLM 等）
都必须遵循此接口。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any

# 导入 QueryResult 类型（Reranker 处理 VectorStore 的查询结果）
from src.libs.vector_store.base_vector_store import QueryResult


class BaseReranker(ABC):
    """
    Reranker 抽象基类
    
    定义所有重排序实现必须遵循的统一接口。
    无论底层使用 None（不改变排序）、CrossEncoder 还是 LLM，
    上层代码都可以通过统一的接口调用。
    """
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[QueryResult],
        trace: Optional[Any] = None
    ) -> List[QueryResult]:
        """
        对候选结果进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表，已按初始分数排序（例如来自 VectorStore.query）
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
                   TraceContext 将在 F1 阶段实现，此处预留接口
        
        Returns:
            List[QueryResult]: 重排序后的结果列表
                              - 结果数量 <= 输入候选数量
                              - 按新的相关性分数降序排列
                              - 每个结果包含更新后的 score（如果重排序算法计算了新分数）
        
        Raises:
            ValueError: 当查询为空或候选列表为空时
            RuntimeError: 当重排序操作失败时（模型错误、网络错误等）
        """
        pass
    
    @abstractmethod
    def get_backend(self) -> str:
        """
        获取当前使用的 backend 名称
        
        Returns:
            str: backend 名称，例如 "none", "cross_encoder", "llm"
        """
        pass
