"""
multimodal_assembler 单元测试
"""
import base64
import json
import tempfile
from pathlib import Path

import pytest

from src.core.response.multimodal_assembler import (
    assemble_content,
    _load_image_index,
    _image_refs_to_content_items,
)
from src.core.response.response_builder import build_mcp_content


def test_load_image_index_missing() -> None:
    """index.json 不存在时返回空 dict"""
    result = _load_image_index("/nonexistent", "coll")
    assert result == {}


def test_load_image_index_valid(tmp_path: Path) -> None:
    """正确加载 index.json"""
    coll_dir = tmp_path / "report"
    coll_dir.mkdir()
    index_data = {
        "collection_name": "report",
        "images": {
            "img_1": {
                "image_id": "img_1",
                "file_path": str(coll_dir / "img_1.png"),
                "mime_type": "image/png",
            },
        },
    }
    (coll_dir / "index.json").write_text(json.dumps(index_data), encoding="utf-8")
    (coll_dir / "img_1.png").write_bytes(b"\x89PNG\x0d\x0a\x1a\x0a")
    result = _load_image_index(str(tmp_path), "report")
    assert "img_1" in result
    assert result["img_1"]["mime_type"] == "image/png"


def test_image_refs_to_content_items(tmp_path: Path) -> None:
    """image_refs 转为 ImageContent dict 列表"""
    coll_dir = tmp_path / "report"
    coll_dir.mkdir()
    img_path = coll_dir / "doc_abc_page_0_img_0.png"
    img_path.write_bytes(b"\x89PNG\x0d\x0a")
    index_data = {
        "images": {
            "doc_abc_page_0_img_0": {
                "file_path": str(img_path),
                "mime_type": "image/png",
            },
        },
    }
    (coll_dir / "index.json").write_text(json.dumps(index_data), encoding="utf-8")
    items = _image_refs_to_content_items(
        ["doc_abc_page_0_img_0"],
        "report",
        str(tmp_path),
        index_data["images"],
    )
    assert len(items) == 1
    assert items[0]["type"] == "image"
    assert items[0]["mimeType"] == "image/png"
    assert base64.b64decode(items[0]["data"]) == b"\x89PNG\x0d\x0a"


def test_assemble_content_no_images() -> None:
    """无 image_refs 时仅返回文本"""
    from src.libs.vector_store.base_vector_store import QueryResult

    results = [
        QueryResult(id="c1", score=0.9, text="hello", metadata={}),
    ]
    content = assemble_content(results, "hello world", "data/images", None)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "hello world"


def test_assemble_content_with_image_refs(tmp_path: Path) -> None:
    """有 image_refs 时追加 ImageContent"""
    from src.libs.vector_store.base_vector_store import QueryResult

    coll_dir = tmp_path / "report"
    coll_dir.mkdir()
    img_path = coll_dir / "img_1.png"
    img_path.write_bytes(b"fake-png-data")
    index_data = {
        "collection_name": "report",
        "images": {
            "img_1": {
                "image_id": "img_1",
                "file_path": str(img_path),
                "mime_type": "image/png",
            },
        },
    }
    (coll_dir / "index.json").write_text(json.dumps(index_data), encoding="utf-8")
    results = [
        QueryResult(id="c1", score=0.9, text="text", metadata={"image_refs": ["img_1"]}),
    ]
    content = assemble_content(
        results, "markdown here", str(tmp_path), "report"
    )
    assert len(content) >= 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert content[1]["data"] == base64.b64encode(b"fake-png-data").decode("ascii")


def test_build_mcp_content_empty() -> None:
    """空结果时返回标准结构（build_mcp_content 已通过 assemble_content 支持 image_refs）"""
    result = build_mcp_content([])
    assert result["content"][0]["type"] == "text"
    assert "未找到相关内容" in result["content"][0]["text"]
    assert "citations" in result["structuredContent"]
    assert result["isError"] is False
