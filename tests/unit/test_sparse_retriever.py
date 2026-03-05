"""
SparseRetriever 单元测试

对已构建索引的 fixtures 语料，验证关键词检索命中预期 chunk_id。
使用 FTS5/SQLite 统一存储。
"""
import os
import tempfile

import pytest

from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.settings import VectorStoreConfig
from src.ingestion.models import Chunk
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.vector_store.base_vector_store import VectorRecord
from src.libs.vector_store.sqlite_store import SQLiteVectorStore


@pytest.fixture
def temp_bm25_dir():
    """临时目录（保留名称兼容）"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    import shutil
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
    """构建 SQLite+FTS5 索引"""
    sqlite_path = os.path.join(temp_bm25_dir, "test.sqlite")
    config = VectorStoreConfig(
        backend="sqlite",
        persist_path="",
        collection_name="test_collection",
        sqlite_path=sqlite_path,
        embedding_dim=16,
    )
    store = SQLiteVectorStore(config, write_fts=True)
    embedding = FakeEmbedding(dimension=16)
    texts = [c.text for c in sample_chunks]
    vectors = embedding.embed(texts)
    records = [
        VectorRecord(id=c.id, vector=vectors[i], text=c.text, metadata=c.metadata)
        for i, c in enumerate(sample_chunks)
    ]
    store.upsert(records, collection_name="test_collection")
    store.close()
    return sqlite_path


class TestSparseRetrieverBasic:
    """基础功能测试"""

    def test_retrieve_returns_query_results(self, indexed_collection) -> None:
        """检索返回 QueryResult 列表"""
        retriever = SparseRetriever(
            sqlite_path=indexed_collection,
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
            sqlite_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve(["python"], top_k=5)

        chunk_ids = [r.id for r in results]
        assert "chunk_1" in chunk_ids or "chunk_3" in chunk_ids
        assert "python" in results[0].text.lower() or "python" in results[0].text

    def test_retrieve_with_keywords_list(self, indexed_collection) -> None:
        """支持关键词列表输入"""
        retriever = SparseRetriever(
            sqlite_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve(["python", "machine"], top_k=5)

        assert len(results) >= 1

    def test_retrieve_with_string_query(self, indexed_collection) -> None:
        """支持字符串输入（jieba 分词）"""
        retriever = SparseRetriever(
            sqlite_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve("python RAG", top_k=5)

        assert len(results) >= 1

    def test_retrieve_empty_keywords_returns_empty(self, indexed_collection) -> None:
        """空关键词返回空列表"""
        retriever = SparseRetriever(
            sqlite_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve([], top_k=5)
        assert results == []

    def test_retrieve_top_k_zero_raises(self, indexed_collection) -> None:
        """top_k <= 0 抛出 ValueError"""
        retriever = SparseRetriever(
            sqlite_path=indexed_collection,
            collection_name="test_collection",
        )
        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            retriever.retrieve(["python"], top_k=0)

    def test_retrieve_no_collection_raises(self, indexed_collection) -> None:
        """未指定 collection 且无默认时抛出 ValueError"""
        retriever = SparseRetriever(sqlite_path=indexed_collection)
        with pytest.raises(ValueError, match="collection_name 未指定"):
            retriever.retrieve(["python"], top_k=5)

    def test_index_exists_true(self, indexed_collection) -> None:
        """index_exists 对已存在索引返回 True"""
        retriever = SparseRetriever(sqlite_path=indexed_collection)
        assert retriever.index_exists("test_collection") is True

    def test_index_exists_false(self, temp_bm25_dir) -> None:
        """index_exists 对不存在索引返回 False"""
        sqlite_path = os.path.join(temp_bm25_dir, "empty.sqlite")
        retriever = SparseRetriever(sqlite_path=sqlite_path)
        assert retriever.index_exists("nonexistent") is False

    def test_index_exists_empty_collection_returns_false(self, temp_bm25_dir) -> None:
        """index_exists 对空字符串返回 False"""
        sqlite_path = os.path.join(temp_bm25_dir, "empty.sqlite")
        retriever = SparseRetriever(sqlite_path=sqlite_path)
        assert retriever.index_exists("") is False


class TestSparseRetrieverWithVectorStore:
    """使用 vector_store.sparse_query 的测试"""

    def test_retrieve_via_vector_store(self, indexed_collection, sample_chunks) -> None:
        """通过 vector_store 的 sparse_query 检索"""
        config = VectorStoreConfig(
            backend="sqlite",
            persist_path="",
            collection_name="test_collection",
            sqlite_path=indexed_collection,
            embedding_dim=16,
        )
        store = SQLiteVectorStore(config, write_fts=True)
        retriever = SparseRetriever(
            vector_store=store,
            sqlite_path=indexed_collection,
            collection_name="test_collection",
        )
        results = retriever.retrieve(["python"], top_k=5)
        store.close()
        assert len(results) >= 1
        assert any(r.id in ("chunk_1", "chunk_3") for r in results)
