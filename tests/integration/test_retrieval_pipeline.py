"""
RetrievalPipeline 集成测试

验证端到端检索流水线：QueryProcessor → HybridSearch → Reranker，能返回 Top-K。
"""
import pytest

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import RerankerOrchestrator
from src.core.query_engine.retrieval_pipeline import RetrievalPipeline
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.settings import RetrievalConfig
from src.libs.reranker.none_reranker import NoneReranker
from src.libs.vector_store.base_vector_store import QueryResult


@pytest.fixture
def retrieval_pipeline(indexed_fixtures):
    """构建完整 RetrievalPipeline"""
    fixtures = indexed_fixtures
    dense = DenseRetriever(
        embedding=fixtures["embedding"],
        vector_store=fixtures["vector_store"],
    )
    sparse = SparseRetriever(
        base_path=fixtures["bm25_path"],
        collection_name=fixtures["collection_name"],
    )
    hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)
    reranker = RerankerOrchestrator(backend=NoneReranker())

    return RetrievalPipeline(
        query_processor=QueryProcessor(),
        hybrid_search=hybrid,
        reranker=reranker,
    )


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
            base_path=fixtures["bm25_path"],
            collection_name=fixtures["collection_name"],
        )
        hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)
        reranker = RerankerOrchestrator(backend=NoneReranker())
        retrieval_config = RetrievalConfig(
            sparse_backend="bm25",
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
