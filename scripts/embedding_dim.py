"""打印当前 embedding 模型的向量维度，供 vector_store.embedding_dim 配置参考。"""
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.core.settings import load_settings

settings = load_settings()
emb = EmbeddingFactory.create(settings)
dim = emb.get_dimension()  # 直接从 embedding 实例获取，无需调用 API
print(f"embedding_dim: {dim}  # {settings.embedding.provider}/{settings.embedding.model}")