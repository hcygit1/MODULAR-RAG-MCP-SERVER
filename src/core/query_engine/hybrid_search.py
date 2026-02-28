"""
HybridSearch 实现

编排 Dense + Sparse 双路检索，经 RRF 融合后返回 Top-K 结果。
"""
from typing import Any, Dict, List, Optional, Union

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import fuse_rrf
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.trace.trace_context import TraceContext
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
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._rrf_k = rrf_k

    def search(
        self,
        query: Union[str, List[str]],
        top_k: int,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        top_k_final: Optional[int] = None,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        执行混合检索：dense + sparse + RRF 融合

        Args:
            query: 查询字符串或关键词列表
            top_k: 返回的 Top-K 数量（当 top_k_dense/sparse/final 未指定时，三者均使用此值）
            top_k_dense: 向量检索候选数，为 None 时使用 top_k
            top_k_sparse: BM25 检索候选数，为 None 时使用 top_k
            top_k_final: 融合后最终返回数，为 None 时使用 top_k
            collection_name: 集合名称，传给 SparseRetriever
            filters: 元数据过滤条件（可选）
            trace: 追踪上下文（可选）

        Returns:
            List[QueryResult]: 按 RRF 融合后排序的 Top-K 结果

        Raises:
            ValueError: 当 query 为空或 top_k <= 0 时
        """
        query_text = query if isinstance(query, str) else " ".join(str(q) for q in query)
        if not query_text or not query_text.strip():
            raise ValueError("query 不能为空")

        k_dense = top_k_dense if top_k_dense is not None else top_k
        k_sparse = top_k_sparse if top_k_sparse is not None else top_k
        k_final = top_k_final if top_k_final is not None else top_k

        if top_k <= 0 or k_dense <= 0 or k_sparse <= 0 or k_final <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: top_k={top_k}, dense={k_dense}, sparse={k_sparse}, final={k_final}")

        _trace: Optional[TraceContext] = trace if isinstance(trace, TraceContext) else None

        # 1. Dense 检索
        coll = collection_name or getattr(self._sparse, "_default_collection", None)
        if _trace:
            with _trace.stage("dense_retrieval", top_k=k_dense):
                dense_results = self._dense.retrieve(
                    query=query, top_k=k_dense, filters=filters, trace=trace,
                    collection_name=coll,
                )
        else:
            dense_results = self._dense.retrieve(
                query=query, top_k=k_dense, filters=filters, trace=trace,
                collection_name=coll,
            )

        # 2. Sparse 检索
        sparse_results: List[QueryResult] = []
        if coll:
            if _trace:
                with _trace.stage("sparse_retrieval", top_k=k_sparse):
                    sparse_results = self._sparse.retrieve(
                        query=query, top_k=k_sparse, collection_name=coll,
                        filters=filters, trace=trace,
                    )
            else:
                sparse_results = self._sparse.retrieve(
                    query=query, top_k=k_sparse, collection_name=coll,
                    filters=filters, trace=trace,
                )

        # 3. RRF 融合
        if _trace:
            with _trace.stage("rrf_fusion", dense_count=len(dense_results), sparse_count=len(sparse_results)):
                fused = fuse_rrf(
                    dense_results=dense_results,
                    sparse_results=sparse_results,
                    k=self._rrf_k,
                )
        else:
            fused = fuse_rrf(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=self._rrf_k,
            )

        return fused[:k_final]
