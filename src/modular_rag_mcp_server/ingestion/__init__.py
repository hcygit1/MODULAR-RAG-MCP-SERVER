"""
Ingestion Pipeline 模块

提供文档摄取、处理、存储的完整流程。
"""
from modular_rag_mcp_server.ingestion.models import Chunk, Document

__all__ = ["Document", "Chunk"]

# IngestionPipeline 从 modular_rag_mcp_server.ingestion.pipeline 导入，避免循环依赖
