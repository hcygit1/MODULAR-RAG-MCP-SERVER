"""
RetrievalPipeline 集成测试

验证端到端检索流水线：QueryProcessor → HybridSearch → Reranker，能返回 Top-K。
"""
from unittest.mock import MagicMock

import pytest

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import ProcessedQuery, QueryProcessor
from src.core.query_engine.reranker import RerankerOrchestrator
from src.core.query_engine.retrieval_pipeline import RetrievalPipeline
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.settings import RetrievalConfig
from src.libs.reranker.none_reranker import NoneReranker
from src.libs.vector_store.base_vector_store import QueryResult




class TestRetrievalPipelineBasic:
    """基础功能测试"""

    def test_retrieve_returns_top_k(
        self,
        retrieval_pipeline,
        indexed_fixtures,
    ) -> None:
        """端到端检索返回 Top-K"""
        results = retrieval_pipeline.retrieve(
            query="python RAG",
            top_k=5,
            collection_name=indexed_fixtures["collection_name"],
        )

        assert len(results) <= 5
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, QueryResult)
            assert r.id
            assert r.text
            assert isinstance(r.metadata, dict)

    def test_retrieve_respects_top_k(
        self,
        retrieval_pipeline,
        indexed_fixtures,
    ) -> None:
        """top_k 限制返回数量"""
        results = retrieval_pipeline.retrieve(
            query="python data",
            top_k=2,
            collection_name=indexed_fixtures["collection_name"],
        )
        assert len(results) <= 2

    def test_retrieve_empty_query_returns_empty(
        self,
        retrieval_pipeline,
    ) -> None:
        """空 query 返回空列表"""
        results = retrieval_pipeline.retrieve(
            query="",
            top_k=5,
            collection_name="test_collection",
        )
        assert results == []

    def test_retrieve_top_k_zero_raises(
        self,
        retrieval_pipeline,
        indexed_fixtures,
    ) -> None:
        """top_k <= 0 抛出 ValueError"""
        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            retrieval_pipeline.retrieve(
                query="python",
                top_k=0,
                collection_name=indexed_fixtures["collection_name"],
            )

    def test_retrieve_with_retrieval_config(
        self,
        indexed_fixtures,
    ) -> None:
        """传入 retrieval_config 时，使用配置中的 top_k_dense/sparse"""
        fixtures = indexed_fixtures
        dense = DenseRetriever(
            embedding=fixtures["embedding"],
            vector_store=fixtures["vector_store"],
        )
        sparse = SparseRetriever(
            vector_store=fixtures["vector_store"],
            sqlite_path=fixtures["sqlite_path"],
            collection_name=fixtures["collection_name"],
        )
        hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)
        reranker = RerankerOrchestrator(backend=NoneReranker())
        retrieval_config = RetrievalConfig(
            sparse_backend="fts5",
            fusion_algorithm="rrf",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_final=10,
        )
        pipeline = RetrievalPipeline(
            query_processor=QueryProcessor(),
            hybrid_search=hybrid,
            reranker=reranker,
            retrieval_config=retrieval_config,
        )

        results = pipeline.retrieve(
            query="python RAG",
            top_k=3,
            collection_name=fixtures["collection_name"],
        )

        assert len(results) <= 3


def _mock_runtime_pipeline(
    config: RetrievalConfig,
    parent_aggregator=None,
):
    query_processor = MagicMock()
    query_processor.process.return_value = ProcessedQuery(
        keywords=["question"],
        filters={},
        original_query="question",
    )
    hybrid_search = MagicMock()
    candidates = [
        QueryResult(
            id=f"chunk-{index}",
            score=1.0 - index / 100,
            text=f"chunk {index}",
            metadata={},
        )
        for index in range(50)
    ]
    hybrid_search.search.return_value = candidates
    reranker = MagicMock()
    reranker.rerank_with_fallback.return_value = (candidates, False)
    pipeline = RetrievalPipeline(
        query_processor=query_processor,
        hybrid_search=hybrid_search,
        reranker=reranker,
        retrieval_config=config,
        parent_aggregator=parent_aggregator,
        collection_name="docs",
    )
    return pipeline, hybrid_search


def test_request_top_k_overrides_configured_final_limit() -> None:
    config = RetrievalConfig(
        sparse_backend="fts5",
        fusion_algorithm="rrf",
        top_k_dense=20,
        top_k_sparse=20,
        top_k_final=10,
    )
    pipeline, hybrid_search = _mock_runtime_pipeline(config)

    results = pipeline.retrieve("question", top_k=3)

    assert len(results) == 3
    assert hybrid_search.search.call_args.kwargs["top_k_final"] == 3
    assert hybrid_search.search.call_args.kwargs["top_k_dense"] == 20
    assert hybrid_search.search.call_args.kwargs["top_k_sparse"] == 20


def test_request_top_k_raises_candidate_budgets_when_needed() -> None:
    config = RetrievalConfig(
        sparse_backend="fts5",
        fusion_algorithm="rrf",
        top_k_dense=10,
        top_k_sparse=20,
        top_k_final=5,
    )
    pipeline, hybrid_search = _mock_runtime_pipeline(config)

    pipeline.retrieve("question", top_k=40)

    kwargs = hybrid_search.search.call_args.kwargs
    assert kwargs["top_k_dense"] == 40
    assert kwargs["top_k_sparse"] == 40
    assert kwargs["top_k_final"] == 40


@pytest.mark.parametrize(
    (
        "parent_top_m",
        "request_top_k",
        "dense_parent",
        "sparse_parent",
        "expected_child_top_k",
        "expected_dense",
        "expected_sparse",
    ),
    [
        (30, 40, 10, 20, 40, 40, 40),
        (30, 3, 10, 50, 30, 30, 50),
    ],
)
def test_parent_mode_preserves_request_limit_and_candidate_floor(
    parent_top_m: int,
    request_top_k: int,
    dense_parent: int,
    sparse_parent: int,
    expected_child_top_k: int,
    expected_dense: int,
    expected_sparse: int,
) -> None:
    config = RetrievalConfig(
        sparse_backend="fts5",
        fusion_algorithm="rrf",
        top_k_dense=10,
        top_k_sparse=20,
        top_k_final=5,
        aggregate_by_parent=True,
        parent_aggregate_top_m=parent_top_m,
        top_k_dense_parent=dense_parent,
        top_k_sparse_parent=sparse_parent,
    )
    parent_aggregator = MagicMock()
    parent_aggregator.aggregate.side_effect = (
        lambda *, child_results, top_k, **_: child_results[:top_k]
    )
    pipeline, hybrid_search = _mock_runtime_pipeline(
        config,
        parent_aggregator=parent_aggregator,
    )

    results = pipeline.retrieve("question", top_k=request_top_k)

    search_kwargs = hybrid_search.search.call_args.kwargs
    assert search_kwargs["top_k"] == expected_child_top_k
    assert search_kwargs["top_k_final"] == expected_child_top_k
    assert search_kwargs["top_k_dense"] == expected_dense
    assert search_kwargs["top_k_sparse"] == expected_sparse
    assert parent_aggregator.aggregate.call_args.kwargs["top_k"] == request_top_k
    assert len(results) == request_top_k
