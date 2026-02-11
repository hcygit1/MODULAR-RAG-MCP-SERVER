"""
SparseRetriever 单元测试

对已构建索引的 fixtures 语料，验证关键词检索命中预期 chunk_id。
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.core.query_engine.sparse_retriever import SparseRetriever
from src.ingestion.models import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.embedding.sparse_encoder import SparseEncoder


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
def indexed_collection(temp_bm25_dir, sample_chunks):
    """构建并保存 BM25 索引"""
    encoder = SparseEncoder()
    sparse_vectors = encoder.encode(sample_chunks)
    indexer = BM25Indexer(base_path=temp_bm25_dir)
    indexer.build(sample_chunks, sparse_vectors, collection_name="test_collection")
    indexer.save()
    return temp_bm25_dir


class TestSparseRetrieverBasic:
    """基础功能测试"""

    def test_retrieve_returns_query_results(self, indexed_collection) -> None:
        """检索返回 QueryResult 列表"""
        retriever = SparseRetriever(
            base_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve(["python"], top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        for r in results:
            assert r.id
            assert isinstance(r.score, float)
            assert isinstance(r.text, str)
            assert isinstance(r.metadata, dict)

    def test_retrieve_hits_expected_chunk_id(self, indexed_collection) -> None:
        """关键词检索命中预期 chunk_id"""
        retriever = SparseRetriever(
            base_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve(["python"], top_k=5)

        chunk_ids = [r.id for r in results]
        assert "chunk_1" in chunk_ids or "chunk_3" in chunk_ids
        assert "python" in results[0].text.lower() or "python" in results[0].text

    def test_retrieve_with_keywords_list(self, indexed_collection) -> None:
        """支持关键词列表输入"""
        retriever = SparseRetriever(
            base_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve(["python", "machine"], top_k=5)

        assert len(results) >= 1
        assert all(r.id for r in results)

    def test_retrieve_with_string_query(self, indexed_collection) -> None:
        """支持字符串输入（按空格分词）"""
        retriever = SparseRetriever(
            base_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve("python RAG", top_k=5)

        assert len(results) >= 1


class TestSparseRetrieverEdgeCases:
    """边界情况测试"""

    def test_empty_keywords_returns_empty(self, indexed_collection) -> None:
        """空关键词返回空列表"""
        retriever = SparseRetriever(
            base_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve([], top_k=5)
        assert results == []

        results = retriever.retrieve("   ", top_k=5)
        assert results == []

    def test_top_k_zero_raises(self, indexed_collection) -> None:
        """top_k <= 0 抛出 ValueError"""
        retriever = SparseRetriever(
            base_path=indexed_collection,
            collection_name="test_collection",
        )
        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            retriever.retrieve(["python"], top_k=0)

    def test_no_collection_raises(self, indexed_collection) -> None:
        """未指定 collection 且无默认时抛出 ValueError"""
        retriever = SparseRetriever(base_path=indexed_collection)
        with pytest.raises(ValueError, match="collection_name 未指定"):
            retriever.retrieve(["python"], top_k=5)

    def test_nonexistent_collection_raises(self, temp_bm25_dir) -> None:
        """不存在的集合加载失败"""
        retriever = SparseRetriever(
            base_path=temp_bm25_dir,
            collection_name="test_collection",
        )
        with pytest.raises(FileNotFoundError, match="索引文件不存在"):
            retriever.retrieve(["python"], top_k=5)


class TestSparseRetrieverFilters:
    """filters 测试"""

    def test_filters_applied(self, indexed_collection) -> None:
        """filters 对结果做 post-filter"""
        retriever = SparseRetriever(
            base_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve(
            ["python"],
            top_k=10,
            filters={"source_path": "/path/to/doc1.pdf"},
        )
        for r in results:
            assert r.metadata.get("source_path") == "/path/to/doc1.pdf"
