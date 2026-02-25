"""
响应构建器

将检索结果（QueryResult 列表）转换为 MCP tools/call 的 content 格式：
- content[0]: 可读 Markdown 文本
- structuredContent.citations: 结构化引用
"""
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import QueryResult

from src.core.response.citation_generator import generate_citations


def _results_to_markdown(results: List["QueryResult"], max_chars_per_chunk: int = 500) -> str:
    """
    将检索结果拼接为 Markdown 文本。

    Args:
        results: 检索结果列表
        max_chars_per_chunk: 每个 chunk 最多显示的字符数

    Returns:
        Markdown 字符串
    """
    if not results:
        return "未找到相关内容。"

    lines: List[str] = []
    for i, r in enumerate(results, 1):
        meta = r.metadata or {}
        source = meta.get("source_path") or meta.get("source_doc_id") or "未知来源"
        if isinstance(source, str) and "/" in source:
            source = source.split("/")[-1]
        text = r.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        lines.append(f"### 片段 {i}（来源：{source}，相关度：{r.score:.2f}）")
        lines.append("")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()


def build_mcp_content(
    results: List["QueryResult"],
    max_chars_per_chunk: int = 500,
) -> Dict[str, Any]:
    """
    构建 MCP tools/call 返回的 content 结构。

    Args:
        results: 检索结果列表
        max_chars_per_chunk: 每个 chunk 在 Markdown 中最多显示的字符数

    Returns:
        MCP content 格式：{ content: [...], structuredContent: { citations: [...] }, isError: False }
    """
    markdown = _results_to_markdown(results, max_chars_per_chunk)
    citations = generate_citations(results)
    return {
        "content": [
            {"type": "text", "text": markdown},
        ],
        "structuredContent": {
            "citations": citations,
        },
        "isError": False,
    }


class ResponseBuilder:
    """响应构建器（类封装）。"""

    def __init__(self, max_chars_per_chunk: int = 500) -> None:
        self._max_chars = max_chars_per_chunk

    def build(self, results: List["QueryResult"]) -> Dict[str, Any]:
        """构建 MCP content。"""
        return build_mcp_content(results, max_chars_per_chunk=self._max_chars)
