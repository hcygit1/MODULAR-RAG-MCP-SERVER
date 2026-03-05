"""
multimodal_assembler 单元测试

Phase C：仅测试 sqlite_path 路径，不再测试文件系统 images_base_path。
"""
import base64
import sqlite3
from pathlib import Path

import pytest

from src.core.response.multimodal_assembler import (
    assemble_content,
    _image_refs_to_content_items,
    _infer_collection_from_sqlite,
)
from src.core.response.response_builder import build_mcp_content


def test_image_refs_to_content_items_sqlite(tmp_path: Path) -> None:
    """image_refs 经 sqlite 转为 ImageContent dict 列表"""
    db_path = tmp_path / "rag.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id TEXT PRIMARY KEY,
            collection_name TEXT NOT NULL,
            image_data BLOB NOT NULL,
            mime_type TEXT DEFAULT 'image/png',
            metadata_json TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        )
    """)
    img_bytes = b"\x89PNG\x0d\x0a\x1a\x0a"
    conn.execute(
        "INSERT INTO images (id, collection_name, image_data, mime_type) VALUES (?, ?, ?, ?)",
        ("doc_abc_page_0_img_0", "report", img_bytes, "image/png"),
    )
    conn.commit()
    conn.close()

    items = _image_refs_to_content_items(
        ["doc_abc_page_0_img_0"],
        "report",
        str(db_path),
    )
    assert len(items) == 1
    assert items[0]["type"] == "image"
    assert items[0]["mimeType"] == "image/png"
    assert base64.b64decode(items[0]["data"]) == img_bytes


def test_infer_collection_from_sqlite(tmp_path: Path) -> None:
    """从 images 表推断 collection_name"""
    db_path = tmp_path / "rag.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id TEXT PRIMARY KEY,
            collection_name TEXT NOT NULL,
            image_data BLOB NOT NULL,
            mime_type TEXT DEFAULT 'image/png',
            metadata_json TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute(
        "INSERT INTO images (id, collection_name, image_data, mime_type) VALUES (?, ?, ?, ?)",
        ("img_x", "my_coll", b"\x89PNG", "image/png"),
    )
    conn.commit()
    conn.close()

    coll = _infer_collection_from_sqlite(str(db_path), "img_x")
    assert coll == "my_coll"
    assert _infer_collection_from_sqlite(str(db_path), "nonexistent") is None


def test_assemble_content_no_images() -> None:
    """无 image_refs 时仅返回文本"""
    from src.libs.vector_store.base_vector_store import QueryResult

    results = [
        QueryResult(id="c1", score=0.9, text="hello", metadata={}),
    ]
    content = assemble_content(results, "hello world", None, None)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "hello world"


def test_assemble_content_no_sqlite_path() -> None:
    """sqlite_path 为空时即使有 image_refs 也不加载图片"""
    from src.libs.vector_store.base_vector_store import QueryResult

    results = [
        QueryResult(id="c1", score=0.9, text="text", metadata={"image_refs": ["img_1"]}),
    ]
    content = assemble_content(results, "markdown here", "report", sqlite_path=None)
    assert len(content) == 1
    assert content[0]["type"] == "text"


def test_assemble_content_sqlite(tmp_path: Path) -> None:
    """sqlite_path 时从 images 表加载图片"""
    from src.libs.vector_store.base_vector_store import QueryResult

    db_path = tmp_path / "rag.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id TEXT PRIMARY KEY,
            collection_name TEXT NOT NULL,
            image_data BLOB NOT NULL,
            mime_type TEXT DEFAULT 'image/png',
            metadata_json TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        )
    """)
    img_bytes = b"\x89PNG\x0d\x0a\x1a\x0a"
    conn.execute(
        "INSERT INTO images (id, collection_name, image_data, mime_type) VALUES (?, ?, ?, ?)",
        ("img_sqlite", "report", img_bytes, "image/png"),
    )
    conn.commit()
    conn.close()

    results = [
        QueryResult(id="c1", score=0.9, text="text", metadata={"image_refs": ["img_sqlite"], "collection_name": "report"}),
    ]
    content = assemble_content(
        results, "markdown", collection_name="report", sqlite_path=str(db_path)
    )
    assert len(content) >= 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert base64.b64decode(content[1]["data"]) == img_bytes


def test_assemble_content_infer_collection(tmp_path: Path) -> None:
    """collection 不在 metadata 时从 images 表推断"""
    from src.libs.vector_store.base_vector_store import QueryResult

    db_path = tmp_path / "rag.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id TEXT PRIMARY KEY,
            collection_name TEXT NOT NULL,
            image_data BLOB NOT NULL,
            mime_type TEXT DEFAULT 'image/png',
            metadata_json TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        )
    """)
    img_bytes = b"\x89PNG"
    conn.execute(
        "INSERT INTO images (id, collection_name, image_data, mime_type) VALUES (?, ?, ?, ?)",
        ("img_infer", "inferred_coll", img_bytes, "image/png"),
    )
    conn.commit()
    conn.close()

    results = [
        QueryResult(id="c1", score=0.9, text="text", metadata={"image_refs": ["img_infer"]}),
    ]
    content = assemble_content(results, "markdown", None, sqlite_path=str(db_path))
    assert len(content) >= 2
    assert content[1]["type"] == "image"
    assert base64.b64decode(content[1]["data"]) == img_bytes


def test_build_mcp_content_empty() -> None:
    """空结果时返回标准结构"""
    result = build_mcp_content([])
    assert result["content"][0]["type"] == "text"
    assert "未找到相关内容" in result["content"][0]["text"]
    assert "citations" in result["structuredContent"]
    assert result["isError"] is False
