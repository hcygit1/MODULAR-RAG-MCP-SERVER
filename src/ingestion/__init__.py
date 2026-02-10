"""
Ingestion Pipeline 模块

提供文档摄取、处理、存储的完整流程。
"""
from src.ingestion.models import Document, Chunk

__all__ = ["Document", "Chunk"]

# IngestionPipeline 从 src.ingestion.pipeline 导入，避免循环依赖
