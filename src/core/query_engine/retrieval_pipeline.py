"""
RetrievalPipeline 编排

串联 QueryProcessor → HybridSearch → Reranker，实现端到端检索流水线。
可选接入 ParentAggregator，支持父子索引场景（heading 切分）。
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import ProcessedQuery, QueryProcessor
from src.core.query_engine.reranker import RerankerOrchestrator
from src.core.trace.trace_context import TraceContext
from src.libs.vector_store.base_vector_store import QueryResult

if TYPE_CHECKING:
    from src.core.query_engine.parent_aggregator import ParentAggregator
    from src.core.settings import RetrievalConfig


class RetrievalPipeline:
    """
    检索流水线编排器

    串联 QueryProcessor、HybridSearch、RerankerOrchestrator，
    对用户 query 执行完整的检索流程并返回 Top-K 结果。
    可选在 rerank 之后接 ParentAggregator，返回父级完整文档。
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        hybrid_search: HybridSearch,
        reranker: RerankerOrchestrator,
        retrieval_config: Optional["RetrievalConfig"] = None,
        parent_aggregator: Optional["ParentAggregator"] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._query_processor = query_processor
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._retrieval_config = retrieval_config
        self._parent_aggregator = parent_aggregator
        self._default_collection = collection_name

    def retrieve(
        self,
        query: str,
        top_k: int,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        执行端到端检索：预处理 → 混合检索 → 精排 → （可选）父聚合

        Args:
            query: 用户查询字符串
            top_k: 返回的 Top-K 数量（父聚合模式下为父级条数）
            collection_name: 集合名称，传给 HybridSearch
            filters: 可选覆盖 ProcessedQuery 的 filters（默认使用 QueryProcessor 解析结果）
            trace: 追踪上下文（可选），为 None 时自动创建

        Returns:
            List[QueryResult]: Top-K 检索结果，含 chunk 文本与 metadata
        """
        if not query or not str(query).strip():
            return []

        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")

        if trace is None:
            trace = TraceContext(operation="retrieval")

        trace.set_metric("query", query)
        trace.set_metric("top_k", top_k)

        with trace.stage("query_processing"):
            processed = self._query_processor.process(query)

        eff_collection = collection_name or self._default_collection
        effective_filters = filters if filters is not None else processed.filters

        # 父聚合模式下，子层需要更多候选才能覆盖足够的父节点
        cfg = self._retrieval_config
        use_parent = (
            self._parent_aggregator is not None
            and cfg is not None
            and getattr(cfg, "aggregate_by_parent", False)
        )

        # 优先使用 cfg.top_k_final（来自 settings.yaml），否则 fallback 到调用方传入的 top_k
        cfg_top_k_final = getattr(cfg, "top_k_final", None) if cfg is not None else None
        effective_top_k_final = cfg_top_k_final if cfg_top_k_final is not None else top_k

        child_top_k = (
            getattr(cfg, "parent_aggregate_top_m", top_k * 5)
            if use_parent
            else effective_top_k_final
        )

        search_kwargs: Dict[str, Any] = {
            "query": processed.original_query,
            "top_k": child_top_k,
            "collection_name": eff_collection,
            "filters": effective_filters,
            "trace": trace,
        }
        if cfg is not None:
            search_kwargs["top_k_dense"] = cfg.top_k_dense
            search_kwargs["top_k_sparse"] = cfg.top_k_sparse
            search_kwargs["top_k_final"] = child_top_k

        hybrid_results = self._hybrid_search.search(**search_kwargs)

        if not hybrid_results:
            trace.set_metric("result_count", 0)
            return []

        results, fallback = self._reranker.rerank_with_fallback(
            query=processed.original_query,
            candidates=hybrid_results,
            trace=trace,
        )

        if use_parent:
            final = self._parent_aggregator.aggregate(
                child_results=results,
                top_k=top_k,
                collection_name=eff_collection or "",
                trace=trace,
            )
            trace.set_metric("aggregation_mode", "parent")
            trace.set_metric("parent_count", len(final))
        else:
            final = results[:effective_top_k_final]

        trace.set_metric("result_count", len(final))
        trace.set_metric("rerank_fallback", fallback)

        return final
