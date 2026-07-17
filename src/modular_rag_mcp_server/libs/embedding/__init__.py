"""
Embedding 模块

提供 Embedding 抽象接口和工厂实现。
"""
from modular_rag_mcp_server.libs.embedding.base_embedding import BaseEmbedding
from modular_rag_mcp_server.libs.embedding.embedding_factory import EmbeddingFactory
from modular_rag_mcp_server.libs.embedding.fake_embedding import FakeEmbedding

# B7.4 阶段实现
from modular_rag_mcp_server.libs.embedding.local_embedding import LocalEmbedding

# B7.3 阶段实现
from modular_rag_mcp_server.libs.embedding.openai_embedding import OpenAIEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "FakeEmbedding",
    "OpenAIEmbedding",
    "LocalEmbedding"
]
