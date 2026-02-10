"""
VectorStore 工厂模块

根据配置创建对应的 VectorStore 实现实例。
支持通过配置文件切换不同的向量数据库后端，无需修改代码。
"""
from typing import Optional

from src.core.settings import Settings
from src.libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """
    VectorStore 工厂类
    
    根据配置动态创建对应的 VectorStore 实现。
    使用工厂模式实现配置驱动的组件选择。
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseVectorStore:
        """
        根据配置创建 VectorStore 实例
        
        Args:
            settings: 配置对象，包含 VectorStore 配置信息
            
        Returns:
            BaseVectorStore: VectorStore 实例
            
        Raises:
            ValueError: 当 backend 不支持或配置不完整时
            NotImplementedError: 当 backend 的实现在 B7 阶段尚未完成时
        """
        config = settings.vector_store
        backend = config.backend.lower()
        
        if backend == "chroma":
            from src.libs.vector_store.chroma_store import ChromaStore
            return ChromaStore(config)
        elif backend == "qdrant":
            from src.libs.vector_store.qdrant_store import QdrantStore
            return QdrantStore(config)
        elif backend == "pinecone":
            # 未来实现
            raise NotImplementedError(
                "Pinecone VectorStore 实现尚未完成。"
                "请先使用其他 backend 或等待实现。"
            )
        else:
            raise ValueError(
                f"不支持的 VectorStore backend: {backend}。"
                f"支持的 backend: chroma, qdrant, pinecone"
            )
    
    @staticmethod
    def create_fake(
        backend: str = "fake",
        collection_name: str = "test_collection"
    ) -> BaseVectorStore:
        """
        创建 Fake VectorStore 实例（用于测试）
        
        Args:
            backend: 后端名称
            collection_name: 集合名称
            
        Returns:
            BaseVectorStore: Fake VectorStore 实例
        """
        from src.libs.vector_store.fake_vector_store import FakeVectorStore
        return FakeVectorStore(backend=backend, collection_name=collection_name)
