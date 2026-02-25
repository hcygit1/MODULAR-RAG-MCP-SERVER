"""
get_document_summary 单元测试

验证对不存在 doc_id 返回规范错误；存在时返回结构化信息。
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.mcp_server.tools.get_document_summary import (
    execute_get_document_summary,
    set_bm25_base_path,
)


@pytest.fixture
def temp_bm25_with_doc(tmp_path: Path):
    """创建含 doc 元数据的 BM25 索引 fixtures"""
    coll_dir = tmp_path / "test_coll"
    coll_dir.mkdir()
    index_file = coll_dir / "index.json"
    chunk_metadata = {
        "doc_abc_chunk_0": {
            "text": "Some chunk text",
            "metadata": {
                "source_doc_id": "doc_abc",
                "source_path": "/path/to/report.pdf",
                "title": "年度报告",
                "summary": "这是一份年度总结",
                "tags": ["报告", "年度"],
        },
            "start_offset": 0,
            "end_offset": 100,
        },
        "doc_abc_chunk_1": {
            "text": "More text",
            "metadata": {"source_doc_id": "doc_abc", "source_path": "/path/to/report.pdf"},
            "start_offset": 100,
            "end_offset": 200,
        },
    }
    index_data = {
        "collection_name": "test_coll",
        "inverted_index": {},
        "chunk_metadata": chunk_metadata,
        "total_chunks": 2,
    }
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)
    return {"base": str(tmp_path), "collection": "test_coll"}


def test_get_document_summary_not_found() -> None:
    """对不存在 doc_id 返回规范错误"""
    with tempfile.TemporaryDirectory() as td:
        set_bm25_base_path(td)
        try:
            result = execute_get_document_summary({"doc_id": "nonexistent_doc"})
            assert result["isError"] is True
            assert "不存在" in result["content"][0]["text"]
            assert result["structuredContent"] == {}
        finally:
            set_bm25_base_path("data/db/bm25")


def test_get_document_summary_empty_doc_id() -> None:
    """doc_id 为空时返回错误"""
    result = execute_get_document_summary({"doc_id": ""})
    assert result["isError"] is True
    assert "不能为空" in result["content"][0]["text"]


def test_get_document_summary_exists_returns_structured(temp_bm25_with_doc) -> None:
    """存在时返回结构化 title/summary/tags"""
    set_bm25_base_path(temp_bm25_with_doc["base"])
    try:
        result = execute_get_document_summary({"doc_id": "doc_abc"})
        assert result["isError"] is False
        assert "structuredContent" in result
        sc = result["structuredContent"]
        assert sc["doc_id"] == "doc_abc"
        assert sc["title"] == "年度报告"
        assert sc["summary"] == "这是一份年度总结"
        assert sc["tags"] == ["报告", "年度"]
        assert "content" in result
        assert "年度报告" in result["content"][0]["text"]
    finally:
        set_bm25_base_path("data/db/bm25")


def test_get_document_summary_with_collection(temp_bm25_with_doc) -> None:
    """指定 collection_name 时仅在对应集合查找"""
    set_bm25_base_path(temp_bm25_with_doc["base"])
    try:
        result = execute_get_document_summary({
            "doc_id": "doc_abc",
            "collection_name": temp_bm25_with_doc["collection"],
        })
        assert result["isError"] is False
        assert result["structuredContent"]["title"] == "年度报告"
    finally:
        set_bm25_base_path("data/db/bm25")


def test_get_document_summary_fallback_title(tmp_path: Path) -> None:
    """无 title 时从 source_path 推断"""
    coll_dir = tmp_path / "fallback_coll"
    coll_dir.mkdir()
    index_data = {
        "collection_name": "fallback_coll",
        "inverted_index": {},
        "chunk_metadata": {
            "doc_xyz_chunk_0": {
                "text": "x",
                "metadata": {"source_doc_id": "doc_xyz", "source_path": "/docs/report.pdf"},
                "start_offset": 0,
                "end_offset": 1,
            },
        },
        "total_chunks": 1,
    }
    with open(coll_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)

    set_bm25_base_path(str(tmp_path))
    try:
        result = execute_get_document_summary({"doc_id": "doc_xyz"})
        assert result["isError"] is False
        assert "report" in result["structuredContent"]["title"]
    finally:
        set_bm25_base_path("data/db/bm25")
