"""
Ingestion Pipeline 模块

提供文档摄取、处理、存储的完整流程。
"""
from src.ingestion.models import Document, Chunk

__all__ = ["Document", "Chunk"]
