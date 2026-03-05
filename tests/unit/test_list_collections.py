"""
list_collections 单元测试

验证从 SQLite chunks 表能返回集合名列表。
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.mcp_server.tools.list_collections import (
    _list_collections_from_sqlite,
    execute_list_collections,
    set_base_path,
)


def test_list_collections_from_sqlite_empty(tmp_path: Path) -> None:
    """空 SQLite（无 chunks 表）返回空列表"""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunks (id TEXT, collection_name TEXT, text TEXT, metadata_json TEXT)"
    )
    conn.commit()
    conn.close()
    assert _list_collections_from_sqlite(str(db_path)) == []


def test_list_collections_from_sqlite_returns_collections(tmp_path: Path) -> None:
    """返回 chunks 表中的 DISTINCT collection_name"""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunks (id TEXT, collection_name TEXT, text TEXT, metadata_json TEXT)"
    )
    conn.execute(
        "INSERT INTO chunks (id, collection_name, text, metadata_json) VALUES (?, ?, ?, ?)",
        ("c1", "report", "text1", None),
    )
    conn.execute(
        "INSERT INTO chunks (id, collection_name, text, metadata_json) VALUES (?, ?, ?, ?)",
        ("c2", "knowledge_base", "text2", None),
    )
    conn.execute(
        "INSERT INTO chunks (id, collection_name, text, metadata_json) VALUES (?, ?, ?, ?)",
        ("c3", "report", "text3", None),
    )
    conn.commit()
    conn.close()
    result = _list_collections_from_sqlite(str(db_path))
    assert sorted(result) == ["knowledge_base", "report"]


def test_list_collections_from_sqlite_nonexistent() -> None:
    """路径不存在时返回空列表（异常被捕获）"""
    result = _list_collections_from_sqlite("/nonexistent/path/xyz.db")
    assert result == []


def test_execute_list_collections_with_injection(tmp_path: Path) -> None:
    """测试注入 sqlite_path 时能返回集合列表"""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunks (id TEXT, collection_name TEXT, text TEXT, metadata_json TEXT)"
    )
    conn.execute(
        "INSERT INTO chunks (id, collection_name, text, metadata_json) VALUES (?, ?, ?, ?)",
        ("c1", "report", "t", None),
    )
    conn.execute(
        "INSERT INTO chunks (id, collection_name, text, metadata_json) VALUES (?, ?, ?, ?)",
        ("c2", "docs", "t", None),
    )
    conn.commit()
    conn.close()

    set_base_path(str(db_path))
    try:
        result = execute_list_collections({})
        assert result["isError"] is False
        assert "collections" in result["structuredContent"]
        collections = result["structuredContent"]["collections"]
        assert set(collections) == {"report", "docs"}
        assert "report" in result["content"][0]["text"]
    finally:
        set_base_path(None)
