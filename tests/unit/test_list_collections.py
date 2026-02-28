"""
list_collections 单元测试

验证对 fixtures 中的目录结构能返回集合名列表。
"""
import tempfile
from pathlib import Path

import pytest

from src.mcp_server.tools.list_collections import (
    execute_list_collections,
    set_base_path,
    _list_collections_from_fs,
)


def test_list_collections_from_fs_empty(tmp_path: Path) -> None:
    """空目录返回空列表"""
    assert _list_collections_from_fs(str(tmp_path)) == []


def test_list_collections_from_fs_returns_subdirs(tmp_path: Path) -> None:
    """返回子目录名作为集合列表"""
    (tmp_path / "report").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "hidden").mkdir()
    (tmp_path / "report" / "file.pdf").touch()
    result = _list_collections_from_fs(str(tmp_path))
    assert sorted(result) == ["docs", "hidden", "report"]


def test_list_collections_from_fs_ignores_files(tmp_path: Path) -> None:
    """忽略文件，只返回目录"""
    (tmp_path / "report").mkdir()
    (tmp_path / "readme.txt").touch()
    result = _list_collections_from_fs(str(tmp_path))
    assert result == ["report"]


def test_list_collections_from_fs_ignores_hidden(tmp_path: Path) -> None:
    """忽略以 . 开头的目录"""
    (tmp_path / "report").mkdir()
    (tmp_path / ".git").mkdir()
    result = _list_collections_from_fs(str(tmp_path))
    assert result == ["report"]


def test_list_collections_from_fs_nonexistent() -> None:
    """路径不存在返回空列表"""
    assert _list_collections_from_fs("/nonexistent/path/xyz") == []


def test_execute_list_collections_with_fixtures(tmp_path: Path) -> None:
    """对 fixtures 中的目录结构能返回集合名列表"""
    (tmp_path / "report").mkdir()
    (tmp_path / "knowledge_base").mkdir()

    set_base_path(str(tmp_path))
    try:
        result = execute_list_collections({})
        assert result["isError"] is False
        assert "collections" in result["structuredContent"]
        collections = result["structuredContent"]["collections"]
        assert set(collections) == {"report", "knowledge_base"}
        assert "report" in result["content"][0]["text"]
    finally:
        set_base_path(None)


def test_execute_list_collections_custom_base_path(tmp_path: Path) -> None:
    """支持通过 arguments 传入 base_path"""
    (tmp_path / "custom_coll").mkdir()
    result = execute_list_collections({"base_path": str(tmp_path)})
    assert result["isError"] is False
    assert result["structuredContent"]["collections"] == ["custom_coll"]
