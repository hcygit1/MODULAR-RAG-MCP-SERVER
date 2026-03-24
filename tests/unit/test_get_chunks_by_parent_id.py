"""
SQLiteVectorStore.get_chunks_by_parent_id 单元测试

覆盖：按父取子块、chunk_index 排序、不同 collection 隔离、无结果返回空列表。
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.core.settings import VectorStoreConfig
from src.libs.vector_store.base_vector_store import VectorRecord
from src.libs.vector_store.sqlite_store import SQLiteVectorStore


def _make_store(tmp_path: Path, dim: int = 4) -> SQLiteVectorStore:
    cfg = VectorStoreConfig(
        backend="sqlite",
        persist_path="",
        collection_name="recipes",
        sqlite_path=str(tmp_path / "rag.sqlite"),
        embedding_dim=dim,
    )
    return SQLiteVectorStore(cfg)


def _record(chunk_id: str, parent_id: str, chunk_index: int, text: str, dim=4) -> VectorRecord:
    return VectorRecord(
        id=chunk_id,
        vector=[0.1] * dim,
        text=text,
        metadata={"parent_id": parent_id, "chunk_index": chunk_index, "source": "test"},
    )


@pytest.fixture
def store(tmp_path):
    s = _make_store(tmp_path)
    yield s
    s.close()


def _insert(store, records, collection="recipes"):
    store.upsert(records, collection_name=collection)


# ---------------------------------------------------------------------------
# 基本功能
# ---------------------------------------------------------------------------

def test_returns_chunks_by_parent_id(store):
    """给定 parent_id，应返回该父下所有子块。"""
    records = [
        _record("c0", "doc_s1", 0, "## 宫保鸡丁"),
        _record("c1", "doc_s1", 1, "### 食材\n鸡肉"),
        _record("c2", "doc_s1", 2, "### 步骤\n翻炒"),
        _record("c3", "doc_s2", 0, "## 鱼香肉丝"),
    ]
    _insert(store, records)

    result = store.get_chunks_by_parent_id("recipes", "doc_s1")
    ids = [r.id for r in result]
    assert set(ids) == {"c0", "c1", "c2"}


def test_sorted_by_chunk_index(store):
    """子块应按 chunk_index 升序排列。"""
    records = [
        _record("c2", "doc_s1", 2, "步骤"),
        _record("c0", "doc_s1", 0, "标题"),
        _record("c1", "doc_s1", 1, "食材"),
    ]
    _insert(store, records)

    result = store.get_chunks_by_parent_id("recipes", "doc_s1")
    indexes = [r.metadata["chunk_index"] for r in result]
    assert indexes == sorted(indexes)


def test_returns_empty_for_unknown_parent(store):
    """不存在的 parent_id 应返回空列表。"""
    _insert(store, [_record("c0", "doc_s1", 0, "内容")])
    result = store.get_chunks_by_parent_id("recipes", "nonexistent_parent")
    assert result == []


def test_collection_isolation(store):
    """不同 collection 的同名 parent_id 不互相干扰。"""
    _insert(store, [_record("c0", "doc_s1", 0, "菜谱内容")], collection="recipes")
    _insert(store, [_record("c1", "doc_s1", 0, "其它内容")], collection="other_col")

    result = store.get_chunks_by_parent_id("recipes", "doc_s1")
    assert len(result) == 1
    assert result[0].id == "c0"


def test_metadata_preserved(store):
    """返回的子块 metadata 应包含原始字段。"""
    _insert(store, [_record("c0", "doc_s1", 0, "内容")])
    result = store.get_chunks_by_parent_id("recipes", "doc_s1")
    assert result[0].metadata.get("parent_id") == "doc_s1"
    assert result[0].metadata.get("source") == "test"


def test_score_is_zero(store):
    """get_chunks_by_parent_id 返回的 QueryResult.score 应为 0.0。"""
    _insert(store, [_record("c0", "doc_s1", 0, "内容")])
    result = store.get_chunks_by_parent_id("recipes", "doc_s1")
    assert result[0].score == 0.0
