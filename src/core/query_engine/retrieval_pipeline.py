"""
RetrievalPipeline 编排

串联 QueryProcessor → HybridSearch → Reranker，实现端到端检索流水线。
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import ProcessedQuery, QueryProcessor
from src.core.query_engine.reranker import RerankerOrchestrator
from src.libs.vector_store.base_vector_store import QueryResult

if TYPE_CHECKING:
    from src.core.settings import RetrievalConfig


class RetrievalPipeline:
    """
    检索流水线编排器

    串联 QueryProcessor、HybridSearch、RerankerOrchestrator，
    对用户 query 执行完整的检索流程并返回 Top-K 结果。
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        hybrid_search: HybridSearch,
        reranker: RerankerOrchestrator,
        retrieval_config: Optional["RetrievalConfig"] = None,
    ) -> None:
        """
        初始化 RetrievalPipeline

        Args:
            query_processor: 查询预处理器
            hybrid_search: 混合检索引擎
            reranker: Reranker 编排器（含 fallback）
            retrieval_config: 检索配置（含 top_k_dense/sparse/final），为 None 时使用 retrieve 传入的 top_k
        """
        self._query_processor = query_processor
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._retrieval_config = retrieval_config

    def retrieve(
        self,
        query: str,
        top_k: int,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        执行端到端检索：预处理 → 混合检索 → 精排

        Args:
            query: 用户查询字符串
            top_k: 返回的 Top-K 数量
            collection_name: 集合名称，传给 HybridSearch
            filters: 可选覆盖 ProcessedQuery 的 filters（默认使用 QueryProcessor 解析结果）
            trace: 追踪上下文（可选）

        Returns:
            List[QueryResult]: Top-K 检索结果，含 chunk 文本与 metadata
        """
        if not query or not str(query).strip():
            return []

        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")

        # 1. 查询预处理
        processed = self._query_processor.process(query)

        # 2. 混合检索（用 original_query，filters 取传入或解析结果）
        effective_filters = filters if filters is not None else processed.filters
        search_kwargs: Dict[str, Any] = {
            "query": processed.original_query,
            "top_k": top_k,
            "collection_name": collection_name,
            "filters": effective_filters,
            "trace": trace,
        }
        if self._retrieval_config is not None:
            search_kwargs["top_k_dense"] = self._retrieval_config.top_k_dense
            search_kwargs["top_k_sparse"] = self._retrieval_config.top_k_sparse
            search_kwargs["top_k_final"] = top_k
        hybrid_results = self._hybrid_search.search(**search_kwargs)

        if not hybrid_results:
            return []

        # 3. 精排（含 fallback）
        results, _ = self._reranker.rerank_with_fallback(
            query=processed.original_query,
            candidates=hybrid_results,
            trace=trace,
        )

        return results[:top_k]
