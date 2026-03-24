"""
ContentEvaluator 单元测试

验证 L2 内容评估：content_non_empty、citations_ok、images_ok、keywords_ok。
"""
import pytest

from src.observability.evaluation.content_evaluator import (
    ContentEvaluator,
    evaluate_content,
)


def test_content_non_empty_pass() -> None:
    """有有效 Markdown 时 content_non_empty=1"""
    mcp = {
        "content": [{"type": "text", "text": "### 片段 1\n\n这是内容"}],
        "structuredContent": {"citations": [{}]},
    }
    r = evaluate_content(mcp, None)
    assert r["content_non_empty"] == 1.0


def test_content_non_empty_fail_empty() -> None:
    """空内容时 content_non_empty=0"""
    mcp = {"content": [{"type": "text", "text": ""}], "structuredContent": {"citations": []}}
    r = evaluate_content(mcp, None)
    assert r["content_non_empty"] == 0.0


def test_content_non_empty_fail_not_found() -> None:
    """未找到相关内容时 content_non_empty=0"""
    mcp = {
        "content": [{"type": "text", "text": "未找到相关内容。"}],
        "structuredContent": {"citations": []},
    }
    r = evaluate_content(mcp, None)
    assert r["content_non_empty"] == 0.0


def test_citations_ok_pass() -> None:
    """citations 数量达标时 citations_ok=1"""
    mcp = {
        "content": [{"type": "text", "text": "x"}],
        "structuredContent": {"citations": [{"a": 1}, {"b": 2}]},
    }
    r = evaluate_content(mcp, {"min_citations": 2})
    assert r["citations_ok"] == 1.0


def test_citations_ok_fail() -> None:
    """citations 不足时 citations_ok=0"""
    mcp = {
        "content": [{"type": "text", "text": "x"}],
        "structuredContent": {"citations": [{}]},
    }
    r = evaluate_content(mcp, {"min_citations": 2})
    assert r["citations_ok"] == 0.0


def test_images_ok_pass() -> None:
    """有图片且 expect_images=true 时 images_ok=1"""
    mcp = {
        "content": [
            {"type": "text", "text": "x"},
            {"type": "image", "data": "base64...", "mimeType": "image/png"},
        ],
        "structuredContent": {"citations": []},
    }
    r = evaluate_content(mcp, {"expect_images": True})
    assert r["images_ok"] == 1.0


def test_images_ok_fail() -> None:
    """expect_images=true 但无图片时 images_ok=0"""
    mcp = {
        "content": [{"type": "text", "text": "x"}],
        "structuredContent": {"citations": []},
    }
    r = evaluate_content(mcp, {"expect_images": True})
    assert r["images_ok"] == 0.0


def test_images_ok_skip_when_not_expected() -> None:
    """未要求图片时 images_ok=1（通过）"""
    mcp = {"content": [{"type": "text", "text": "x"}], "structuredContent": {"citations": []}}
    r = evaluate_content(mcp, {})
    assert r["images_ok"] == 1.0


def test_keywords_ok_pass() -> None:
    """关键词全命中时 keywords_ok=1"""
    mcp = {
        "content": [{"type": "text", "text": "系统架构图包含模块A和模块B"}],
        "structuredContent": {"citations": []},
    }
    r = evaluate_content(mcp, {"keywords_in_markdown": ["架构", "系统"]})
    assert r["keywords_ok"] == 1.0


def test_keywords_ok_fail() -> None:
    """关键词未全命中时 keywords_ok=0"""
    mcp = {
        "content": [{"type": "text", "text": "只有架构"}],
        "structuredContent": {"citations": []},
    }
    r = evaluate_content(mcp, {"keywords_in_markdown": ["架构", "系统"]})
    assert r["keywords_ok"] == 0.0


def test_content_evaluator_class() -> None:
    """ContentEvaluator 类接口"""
    ev = ContentEvaluator()
    assert ev.get_backend() == "content"
    mcp = {"content": [{"type": "text", "text": "x"}], "structuredContent": {"citations": []}}
    r = ev.evaluate(mcp, None)
    assert "content_non_empty" in r
    assert r["content_non_empty"] == 1.0
