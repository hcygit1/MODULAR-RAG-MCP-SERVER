"""
引用生成器

从 QueryResult 列表生成结构化引用（source, page, chunk_id, score），
供 MCP tools/call 的 structuredContent.citations 使用。
"""
import os
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import QueryResult


def generate_citations(results: List["QueryResult"]) -> List[Dict[str, Any]]:
    """
    从检索结果生成结构化引用列表。

    Args:
        results: RetrievalPipeline.retrieve() 返回的 QueryResult 列表

    Returns:
        引用列表，每项含 source, page, chunk_id, score
    """
    citations: List[Dict[str, Any]] = []
    for r in results:
        meta = r.metadata or {}
        source = meta.get("source_path") or meta.get("source_doc_id") or meta.get("source") or "unknown"
        if isinstance(source, str) and ("/" in source or "\\" in source):
            source = os.path.basename(source)
        page = meta.get("page")
        if page is None:
            page = meta.get("chunk_index")
        citations.append({
            "source": str(source),
            "page": page,
            "chunk_id": r.id,
            "score": round(r.score, 4),
        })
    return citations


class CitationGenerator:
    """引用生成器（类封装，便于扩展）。"""

    @staticmethod
    def from_results(results: List["QueryResult"]) -> List[Dict[str, Any]]:
        """从 QueryResult 列表生成引用。"""
        return generate_citations(results)
