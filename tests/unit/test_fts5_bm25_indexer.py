"""
FTS5BM25Indexer 单元测试

验证 FTS5 全文索引的 build、merge、query、index_exists。
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.ingestion.models import Chunk
from src.ingestion.storage.fts5_bm25_indexer import FTS5BM25Indexer
from src.ingestion.embedding.sparse_encoder import SparseEncoder


@pytest.fixture
def temp_db():
    """创建临时 SQLite 数据库"""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_chunks():
    """测试用 chunks"""
    return [
        Chunk(id="chunk_1", text="Python is a programming language", metadata={"k": 1}),
        Chunk(
            id="chunk_2",
            text="Machine learning is a subset of artificial intelligence",
            metadata={"k": 2},
        ),
        Chunk(
            id="chunk_3",
            text="Python and RAG for retrieval augmented generation",
            metadata={"k": 3},
        ),
    ]


@pytest.fixture
def sample_sparse_vectors(sample_chunks):
    encoder = SparseEncoder()
    return encoder.encode(sample_chunks)


class TestFTS5BM25IndexerBasic:
    """基础功能测试"""

    def test_build_and_query(self, temp_db, sample_chunks, sample_sparse_vectors):
        """build 后能 query"""
        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        idx.build(sample_chunks, sample_sparse_vectors, collection_name="test")
        idx.load("test")
        results = idx.query(["python"], top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        for cid, score in results:
            assert isinstance(cid, str)
            assert isinstance(score, (int, float))
        idx.close()

    def test_merge_incremental(self, temp_db):
        """merge 增量合并"""
        encoder = SparseEncoder()
        batch1 = [
            Chunk(id="c1", text="Python programming", metadata={}),
            Chunk(id="c2", text="Java programming", metadata={}),
        ]
        batch2 = [
            Chunk(id="c3", text="Python machine learning", metadata={}),
        ]
        sv1 = encoder.encode(batch1)
        sv2 = encoder.encode(batch2)

        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        assert idx.index_exists("merge_test") is False

        idx.merge(batch1, sv1, collection_name="merge_test")
        idx.load("merge_test")
        assert idx.index_exists("merge_test") is True
        r1 = idx.query(["python"], top_k=5)
        assert len(r1) >= 1

        idx.merge(batch2, sv2, collection_name="merge_test")
        r2 = idx.query(["python", "learning"], top_k=5)
        assert len(r2) >= 1
        cids = [cid for cid, _ in r2]
        assert "c3" in cids or "c1" in cids
        idx.close()

    def test_merge_overwrite_same_chunk_id(self, temp_db):
        """merge 时同一 chunk_id 被覆盖"""
        encoder = SparseEncoder()
        c1 = Chunk(id="c1", text="Original Python text", metadata={})
        c2 = Chunk(id="c1", text="Updated Python text", metadata={})
        sv1 = encoder.encode([c1])
        sv2 = encoder.encode([c2])

        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        idx.merge([c1], sv1, collection_name="ow")
        idx.merge([c2], sv2, collection_name="ow")
        idx.load("ow")
        results = idx.query(["updated"], top_k=3)
        assert len(results) >= 1
        assert results[0][0] == "c1"
        idx.close()

    def test_query_empty_terms(self, temp_db, sample_chunks, sample_sparse_vectors):
        """空查询词返回空列表"""
        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        idx.build(sample_chunks, sample_sparse_vectors, collection_name="test")
        idx.load("test")
        assert idx.query([], top_k=3) == []
        idx.close()

    def test_query_nonexistent_terms(self, temp_db, sample_chunks, sample_sparse_vectors):
        """不存在的词返回空或少量结果"""
        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        idx.build(sample_chunks, sample_sparse_vectors, collection_name="test")
        idx.load("test")
        r = idx.query(["xyznonexistent123"], top_k=3)
        assert len(r) == 0
        idx.close()

    def test_save_noop(self, temp_db, sample_chunks, sample_sparse_vectors):
        """save 为 no-op，不报错"""
        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        idx.build(sample_chunks, sample_sparse_vectors, collection_name="test")
        idx.save()
        idx.close()

    def test_chinese_tokenization(self, temp_db):
        """中文分词正确索引与查询"""
        chunks = [
            Chunk(id="zh1", text="自然语言处理是人工智能的重要分支", metadata={}),
        ]
        encoder = SparseEncoder()
        sv = encoder.encode(chunks)

        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        idx.build(chunks, sv, collection_name="zh")
        idx.load("zh")
        results = idx.query(["自然语言", "人工智能"], top_k=3)
        assert len(results) >= 1
        idx.close()

    def test_init_empty_path_raises(self):
        with pytest.raises(ValueError, match="sqlite_path 不能为空"):
            FTS5BM25Indexer(sqlite_path="")

    def test_build_empty_chunks_raises(self, temp_db):
        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        with pytest.raises(ValueError, match="chunks 列表不能为空"):
            idx.build([], [{}], "test")
        idx.close()

    def test_query_without_load_raises(self, temp_db):
        idx = FTS5BM25Indexer(sqlite_path=temp_db)
        with pytest.raises(RuntimeError, match="索引未加载"):
            idx.query(["python"], top_k=3)
        idx.close()
