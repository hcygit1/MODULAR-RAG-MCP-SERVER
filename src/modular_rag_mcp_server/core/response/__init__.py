"""
Core 响应构建模块

提供 Markdown 响应构建、引用生成等能力。
"""

from modular_rag_mcp_server.core.response.citation_generator import (
    CitationGenerator,
    generate_citations,
)
from modular_rag_mcp_server.core.response.response_builder import ResponseBuilder, build_mcp_content

__all__ = [
    "CitationGenerator",
    "ResponseBuilder",
    "build_mcp_content",
    "generate_citations",
]
