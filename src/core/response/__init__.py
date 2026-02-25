"""
Core 响应构建模块

提供 Markdown 响应构建、引用生成等能力。
"""

from src.core.response.citation_generator import CitationGenerator, generate_citations
from src.core.response.response_builder import ResponseBuilder, build_mcp_content

__all__ = [
    "CitationGenerator",
    "ResponseBuilder",
    "build_mcp_content",
    "generate_citations",
]
