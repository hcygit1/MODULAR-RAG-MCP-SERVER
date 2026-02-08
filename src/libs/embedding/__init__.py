"""
Embedding 模块

提供 Embedding 抽象接口和工厂实现。
"""
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.fake_embedding import FakeEmbedding

# B7.3 阶段实现
from src.libs.embedding.openai_embedding import OpenAIEmbedding

# B7.4 阶段实现
from src.libs.embedding.local_embedding import LocalEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "FakeEmbedding",
    "OpenAIEmbedding",
    "LocalEmbedding"
]
