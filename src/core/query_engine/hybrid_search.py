"""
HybridSearch 实现

编排 Dense + Sparse 双路检索，经 RRF 融合后返回 Top-K 结果。
"""
from typing import Any, Dict, List, Optional, Union

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import fuse_rrf
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.libs.vector_store.base_vector_store import QueryResult


class HybridSearch:
    """
    Hybrid Search 编排器

    串联 DenseRetriever、SparseRetriever 与 RRF 融合，
    对同一 query 执行双路检索并返回统一排序的 Top-K 结果。
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: float = 60.0,
    ) -> None:
        """
        初始化 HybridSearch

        Args:
            dense_retriever: 稠密向量检索器
            sparse_retriever: 稀疏检索器（BM25）
            rrf_k: RRF 融合的平滑常数，默认 60
        """
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._rrf_k = rrf_k

    def search(
        self,
        query: Union[str, List[str]],
        top_k: int,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        执行混合检索：dense + sparse + RRF 融合

        Args:
            query: 查询字符串或关键词列表
            top_k: 返回的 Top-K 数量
            collection_name: 集合名称，传给 SparseRetriever（Dense 的 collection 由 VectorStore 绑定）
            filters: 元数据过滤条件（可选），传给 Dense 与 Sparse
            trace: 追踪上下文（可选）

        Returns:
            List[QueryResult]: 按 RRF 融合后排序的 Top-K 结果，含 chunk 文本与 metadata

        Raises:
            ValueError: 当 query 为空或 top_k <= 0 时
        """
        query_text = query if isinstance(query, str) else " ".join(str(q) for q in query)
        if not query_text or not query_text.strip():
            raise ValueError("query 不能为空")

        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")

        # 1. Dense 检索
        dense_results = self._dense.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            trace=trace,
        )

        # 2. Sparse 检索（需 collection_name 或 sparse 构造时指定默认集合）
        sparse_results: List[QueryResult] = []
        coll = collection_name or getattr(self._sparse, "_default_collection", None)
        if coll:
            sparse_results = self._sparse.retrieve(
                query=query,
                top_k=top_k,
                collection_name=coll,
                filters=filters,
                trace=trace,
            )

        # 3. RRF 融合
        fused = fuse_rrf(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=self._rrf_k,
        )

        return fused[:top_k]
