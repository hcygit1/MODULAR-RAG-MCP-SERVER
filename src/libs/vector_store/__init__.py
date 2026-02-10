"""
VectorStore 模块

提供向量存储抽象接口和工厂实现。
"""
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult
)
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.libs.vector_store.fake_vector_store import FakeVectorStore

from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.qdrant_store import QdrantStore

__all__ = [
    "BaseVectorStore",
    "VectorRecord",
    "QueryResult",
    "VectorStoreFactory",
    "FakeVectorStore",
    "ChromaStore",
    "QdrantStore",
]
