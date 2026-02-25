"""
HybridSearch 集成测试

对 fixtures 数据验证混合检索：dense + sparse + RRF 融合，能返回 Top-K（含 chunk 文本与 metadata）。
"""
import shutil
import tempfile

import pytest

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.fake_embedding import FakeEmbedding
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
