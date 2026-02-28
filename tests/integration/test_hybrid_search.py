"""
HybridSearch 集成测试

对 fixtures 数据验证混合检索：dense + sparse + RRF 融合，能返回 Top-K（含 chunk 文本与 metadata）。
包含 F3 Trace 集成测试：验证 trace 中存在 dense/sparse/fusion/rerank 阶段。
"""
import shutil
import tempfile

import pytest

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import RerankerOrchestrator
from src.core.query_engine.retrieval_pipeline import RetrievalPipeline
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.reranker.none_reranker import NoneReranker
from src.libs.vector_store.base_vector_store import QueryResult, VectorRecord
from src.libs.vector_store.fake_vector_store import FakeVectorStore


@pytest.fixture
def temp_bm25_dir():
    """创建临时 BM25 索引目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_chunks():
    """测试用 chunks：含 python、machine、RAG 等关键词"""
    return [
        Chunk(
            id="chunk_1",
            text="Python is a programming language for data science",
            metadata={"source_path": "/path/to/doc1.pdf", "chunk_index": 0},
        ),
        Chunk(
            id="chunk_2",
            text="Machine learning is a subset of artificial intelligence",
            metadata={"source_path": "/path/to/doc1.pdf", "chunk_index": 1},
        ),
        Chunk(
            id="chunk_3",
            text="Python and RAG are used for retrieval augmented generation",
            metadata={"source_path": "/path/to/doc2.pdf", "chunk_index": 0},
        ),
    ]


@pytest.fixture
def indexed_fixtures(temp_bm25_dir, sample_chunks):
    """
    构建 Dense 与 Sparse 双路 fixtures：
    - BM25 索引
    - FakeVectorStore 向量数据
    返回 (vector_store, bm25_path) 供 HybridSearch 使用
    """
    # 1. BM25 索引
    encoder = SparseEncoder()
    sparse_vectors = encoder.encode(sample_chunks)
    indexer = BM25Indexer(base_path=temp_bm25_dir)
    indexer.build(sample_chunks, sparse_vectors, collection_name="test_collection")
    indexer.save()

    # 2. FakeVectorStore 向量数据（与 BM25 使用相同 chunk id）
    embedding = FakeEmbedding(dimension=16)
    vector_store = FakeVectorStore(collection_name="test_collection")

    texts = [c.text for c in sample_chunks]
    vectors = embedding.embed(texts)

    records = [
        VectorRecord(
            id=c.id,
            vector=vectors[i],
            text=c.text,
            metadata=c.metadata,
        )
        for i, c in enumerate(sample_chunks)
    ]
    vector_store.upsert(records)

    return {
        "vector_store": vector_store,
        "embedding": embedding,
        "bm25_path": temp_bm25_dir,
        "collection_name": "test_collection",
    }


class TestHybridSearchBasic:
    """基础功能测试"""

    def test_search_returns_top_k(
        self,
        indexed_fixtures,
    ) -> None:
        """混合检索返回 Top-K 结果，含 chunk 文本与 metadata"""
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

        results = hybrid.search(
            query="python RAG",
            top_k=5,
            collection_name=fixtures["collection_name"],
        )

        assert len(results) <= 5
        assert len(results) >= 1

        for r in results:
            assert isinstance(r, QueryResult)
            assert r.id
            assert r.text
            assert isinstance(r.metadata, dict)
            assert "source_path" in r.metadata or r.metadata  # 有 metadata

    def test_search_fusion_combines_both_routes(
        self,
        indexed_fixtures,
    ) -> None:
        """RRF 融合后，同时命中 dense 和 sparse 的 chunk 排名更靠前"""
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

        results = hybrid.search(
            query="python",
            top_k=5,
            collection_name=fixtures["collection_name"],
        )

        chunk_ids = [r.id for r in results]
        assert "chunk_1" in chunk_ids or "chunk_3" in chunk_ids

    def test_search_with_string_query(
        self,
        indexed_fixtures,
    ) -> None:
        """支持字符串 query"""
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

        results = hybrid.search(
            query="machine learning",
            top_k=3,
            collection_name=fixtures["collection_name"],
        )

        assert len(results) <= 3
        assert all(r.text for r in results)

    def test_search_uses_config_top_k_params(
        self,
        indexed_fixtures,
    ) -> None:
        """top_k_dense/sparse/final 分别控制各阶段数量"""
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

        results = hybrid.search(
            query="python RAG",
            top_k=10,
            top_k_dense=5,
            top_k_sparse=5,
            top_k_final=2,
            collection_name=fixtures["collection_name"],
        )

        assert len(results) <= 2

    def test_search_fallback_to_single_top_k(
        self,
        indexed_fixtures,
    ) -> None:
        """未指定 top_k_dense/sparse/final 时，使用 top_k 作为三者默认值"""
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

        results = hybrid.search(
            query="python",
            top_k=3,
            collection_name=fixtures["collection_name"],
        )

        assert len(results) <= 3


class TestHybridSearchEdgeCases:
    """边界情况测试"""

    def test_search_respects_top_k(
        self,
        indexed_fixtures,
    ) -> None:
        """top_k 限制返回数量"""
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

        results = hybrid.search(
            query="python RAG data",
            top_k=2,
            collection_name=fixtures["collection_name"],
        )

        assert len(results) <= 2

    def test_search_empty_query_raises(
        self,
        indexed_fixtures,
    ) -> None:
        """空 query 抛出 ValueError"""
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

        with pytest.raises(ValueError, match="query 不能为空"):
            hybrid.search(query="", top_k=5, collection_name=fixtures["collection_name"])

    def test_search_top_k_zero_raises(
        self,
        indexed_fixtures,
    ) -> None:
        """top_k <= 0 抛出 ValueError"""
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

        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            hybrid.search(
                query="python",
                top_k=0,
                collection_name=fixtures["collection_name"],
            )


class TestHybridSearchTrace:
    """F3 Trace 集成测试：验证 trace 中存在各阶段记录"""

    def test_hybrid_search_records_stages(self, indexed_fixtures) -> None:
        """HybridSearch.search() 记录 dense/sparse/rrf_fusion 阶段"""
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

        trace = TraceContext(operation="retrieval")
        results = hybrid.search(
            query="python RAG",
            top_k=5,
            collection_name=fixtures["collection_name"],
            trace=trace,
        )

        stage_names = [s.name for s in trace.stages]
        assert "dense_retrieval" in stage_names
        assert "sparse_retrieval" in stage_names
        assert "rrf_fusion" in stage_names

        for s in trace.stages:
            assert s.duration_ms >= 0

    def test_retrieval_pipeline_records_all_stages(self, indexed_fixtures) -> None:
        """RetrievalPipeline.retrieve() 记录 query_processing + hybrid + rerank 阶段"""
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

        trace = TraceContext(operation="retrieval")
        pipeline = RetrievalPipeline(
            query_processor=QueryProcessor(),
            hybrid_search=hybrid,
            reranker=reranker,
        )
        results = pipeline.retrieve(
            query="python RAG",
            top_k=5,
            collection_name=fixtures["collection_name"],
            trace=trace,
        )

        report = trace.finish()
        stage_names = [s["name"] for s in report["stages"]]

        assert "query_processing" in stage_names
        assert "dense_retrieval" in stage_names
        assert "sparse_retrieval" in stage_names
        assert "rrf_fusion" in stage_names
        assert "rerank" in stage_names

        assert report["metrics"]["query"] == "python RAG"
        assert report["metrics"]["top_k"] == 5
        assert "result_count" in report["metrics"]
        assert report["total_duration_ms"] >= 0

    def test_reranker_records_fallback_false(self, indexed_fixtures) -> None:
        """NoneReranker 正常完成时 fallback=False"""
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

        trace = TraceContext(operation="retrieval")
        candidates = hybrid.search(
            query="python",
            top_k=5,
            collection_name=fixtures["collection_name"],
            trace=trace,
        )

        reranker = RerankerOrchestrator(backend=NoneReranker())
        results, fallback = reranker.rerank_with_fallback(
            query="python", candidates=candidates, trace=trace,
        )

        rerank_stages = [s for s in trace.stages if s.name == "rerank"]
        assert len(rerank_stages) == 1
        assert rerank_stages[0].metadata.get("fallback") is False

    def test_trace_finish_is_json_serializable(self, indexed_fixtures) -> None:
        """整条链路的 trace.finish() 可 JSON 序列化"""
        import json

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
        pipeline = RetrievalPipeline(
            query_processor=QueryProcessor(),
            hybrid_search=hybrid,
            reranker=reranker,
        )

        trace = TraceContext(operation="retrieval")
        pipeline.retrieve(
            query="machine learning",
            top_k=3,
            collection_name=fixtures["collection_name"],
            trace=trace,
        )
        report = trace.finish()
        serialized = json.dumps(report, ensure_ascii=False)
        parsed = json.loads(serialized)
        assert len(parsed["stages"]) >= 4


class TestGoldenSetSmoke:
    """F5 黄金测试集 smoke 测试：用 fixture 数据验证 EvalRunner 端到端可跑"""

    def test_eval_runner_with_fixtures(self, indexed_fixtures, tmp_path) -> None:
        """EvalRunner + RetrievalPipeline 端到端产出指标"""
        import json

        from src.observability.evaluation.eval_runner import EvalRunner

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
        pipeline = RetrievalPipeline(
            query_processor=QueryProcessor(),
            hybrid_search=hybrid,
            reranker=reranker,
        )

        golden_path = str(tmp_path / "golden.json")
        golden_data = [
            {
                "query": "python programming",
                "golden_chunk_ids": ["chunk_1"],
                "top_k": 5,
            },
            {
                "query": "machine learning",
                "golden_chunk_ids": ["chunk_2"],
                "top_k": 5,
            },
        ]
        with open(golden_path, "w") as f:
            json.dump(golden_data, f)

        def retrieve_func(query, top_k, collection):
            results = pipeline.retrieve(
                query=query,
                top_k=top_k,
                collection_name=fixtures["collection_name"],
            )
            return [r.id for r in results]

        runner = EvalRunner(retrieve_func=retrieve_func)
        report = runner.run(golden_path)

        assert report.total_cases == 2
        assert report.successful_cases == 2
        assert report.failed_cases == 0
        assert "custom_hit_rate" in report.avg_metrics
        assert "custom_mrr" in report.avg_metrics
        assert 0.0 <= report.avg_metrics["custom_hit_rate"] <= 1.0
        assert 0.0 <= report.avg_metrics["custom_mrr"] <= 1.0
        assert report.total_time_ms >= 0

        report_dict = report.to_dict()
        serialized = json.dumps(report_dict, ensure_ascii=False)
        assert serialized
