"""
Loader 模块

提供文件加载功能。

完整性检查仍从此处导出兼容入口，但实现归属于 ingestion.integrity。
"""
from modular_rag_mcp_server.ingestion.integrity import (
    FileIntegrityChecker,
    compute_sha256,
    mark_success,
    should_skip,
)
from modular_rag_mcp_server.libs.loader.base_loader import BaseLoader
from modular_rag_mcp_server.libs.loader.pdf_loader import PdfLoader

__all__ = [
    "FileIntegrityChecker",
    "compute_sha256",
    "should_skip",
    "mark_success",
    "BaseLoader",
    "PdfLoader",
]
