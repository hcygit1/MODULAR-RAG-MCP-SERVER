"""
L2 内容评估器

对 MCP content（Markdown + citations + images）做规则检查，
产出 content_non_empty、citations_ok、images_ok、keywords_ok 等指标。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

EMPTY_MESSAGE = "未找到相关内容。"


def _extract_markdown(content_list: List[Dict[str, Any]]) -> str:
    """从 content 列表中提取 Markdown 文本。"""
    if not content_list:
        return ""
    texts: List[str] = []
    for item in content_list:
        if isinstance(item, dict) and item.get("type") == "text":
            t = item.get("text")
            if isinstance(t, str):
                texts.append(t)
    return "".join(texts).strip()


def _has_images(content_list: List[Dict[str, Any]]) -> bool:
    """检查 content 中是否包含图片。"""
    if not content_list:
        return False
    for item in content_list:
        if isinstance(item, dict) and item.get("type") == "image":
            return True
    return False


def _get_citations_count(mcp_content: Dict[str, Any]) -> int:
    """获取 citations 数量。"""
    structured = mcp_content.get("structuredContent") or {}
    citations = structured.get("citations")
    if not isinstance(citations, list):
        return 0
    return len(citations)


def evaluate_content(
    mcp_content: Dict[str, Any],
    expected_checks: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    对 MCP content 做 L2 规则评估。

    Args:
        mcp_content: build_mcp_content() 返回的 dict
        expected_checks: 可选，来自 golden case 的 expected_content_checks

    Returns:
        {"content_non_empty": 0|1, "citations_ok": 0|1, "images_ok": 0|1, "keywords_ok": 0|1}
    """
    expected_checks = expected_checks or {}
    content_list = mcp_content.get("content")
    if not isinstance(content_list, list):
        content_list = []

    markdown = _extract_markdown(content_list)
    citations_count = _get_citations_count(mcp_content)
    has_images = _has_images(content_list)

    # content_non_empty
    content_non_empty = 1.0 if (markdown and markdown != EMPTY_MESSAGE) else 0.0

    # citations_ok
    min_citations = expected_checks.get("min_citations", 0)
    citations_ok = 1.0 if citations_count >= min_citations else 0.0

    # images_ok
    expect_images = expected_checks.get("expect_images", False)
    if expect_images:
        images_ok = 1.0 if has_images else 0.0
    else:
        images_ok = 1.0  # 未要求则视为通过

    # keywords_ok
    keywords = expected_checks.get("keywords_in_markdown")
    if isinstance(keywords, list) and keywords:
        keywords_ok = 1.0 if all(kw in markdown for kw in keywords) else 0.0
    else:
        keywords_ok = 1.0  # 未指定则视为通过

    return {
        "content_non_empty": content_non_empty,
        "citations_ok": citations_ok,
        "images_ok": images_ok,
        "keywords_ok": keywords_ok,
    }


class ContentEvaluator:
    """L2 内容评估器（类封装）。"""

    def get_backend(self) -> str:
        return "content"

    def evaluate(
        self,
        mcp_content: Dict[str, Any],
        expected_checks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """评估 MCP content，返回 L2 指标。"""
        return evaluate_content(mcp_content, expected_checks)
