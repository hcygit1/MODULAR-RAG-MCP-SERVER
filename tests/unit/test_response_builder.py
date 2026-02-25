"""Response builder 单元测试"""
import pytest

from src.core.response.response_builder import ResponseBuilder, build_mcp_content
from src.libs.vector_store.base_vector_store import QueryResult


def test_build_mcp_content_empty() -> None:
    """空结果返回未找到内容"""
    result = build_mcp_content([])
    assert result["content"][0]["type"] == "text"
    assert "未找到" in result["content"][0]["text"]
    assert result["structuredContent"]["citations"] == []
    assert result["isError"] is False


def test_build_mcp_content_with_results() -> None:
    """有结果时返回 Markdown + citations"""
    results = [
        QueryResult(
            id="c1",
            score=0.9,
            text="Python is a programming language.",
            metadata={"source_path": "/x.pdf", "page": 1},
        ),
    ]
    result = build_mcp_content(results)
    assert result["content"][0]["type"] == "text"
    assert "Python" in result["content"][0]["text"]
    assert result["structuredContent"]["citations"][0]["chunk_id"] == "c1"
    assert result["structuredContent"]["citations"][0]["score"] == 0.9


class TestResponseBuilder:
    def test_build(self) -> None:
        """ResponseBuilder.build"""
        builder = ResponseBuilder(max_chars_per_chunk=10)
        r = QueryResult(id="c1", score=0.5, text="hello world", metadata={})
        out = builder.build([r])
        assert "content" in out
        assert "hello" in out["content"][0]["text"]
