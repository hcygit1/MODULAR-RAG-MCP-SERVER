"""
VectorStore 模块

提供向量存储抽象接口和工厂实现。
"""
from modular_rag_mcp_server.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    QueryResult,
    VectorRecord,
)
from modular_rag_mcp_server.libs.vector_store.chroma_store import ChromaStore
from modular_rag_mcp_server.libs.vector_store.fake_vector_store import FakeVectorStore
from modular_rag_mcp_server.libs.vector_store.fts5_bm25_indexer import FTS5BM25Indexer
from modular_rag_mcp_server.libs.vector_store.qdrant_store import QdrantStore
from modular_rag_mcp_server.libs.vector_store.sqlite_store import SQLiteVectorStore
from modular_rag_mcp_server.libs.vector_store.vector_store_factory import VectorStoreFactory

__all__ = [
    "BaseVectorStore",
    "VectorRecord",
    "QueryResult",
    "VectorStoreFactory",
    "FakeVectorStore",
    "FTS5BM25Indexer",
    "ChromaStore",
    "QdrantStore",
    "SQLiteVectorStore",
]
