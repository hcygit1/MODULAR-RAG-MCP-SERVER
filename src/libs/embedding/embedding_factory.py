"""
Embedding 工厂模块

根据配置创建对应的 Embedding 实现实例。
支持通过配置文件切换不同的 Embedding provider，无需修改代码。
"""
from typing import Optional

from src.core.settings import Settings
from src.libs.embedding.base_embedding import BaseEmbedding


class EmbeddingFactory:
    """
    Embedding 工厂类
    
    根据配置动态创建对应的 Embedding 实现。
    使用工厂模式实现配置驱动的组件选择。
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseEmbedding:
        """
        根据配置创建 Embedding 实例
        
        Args:
            settings: 配置对象，包含 Embedding 配置信息
            
        Returns:
            BaseEmbedding: Embedding 实例
            
        Raises:
            ValueError: 当 provider 不支持或配置不完整时
            NotImplementedError: 当 provider 的实现在 B7 阶段尚未完成时
        """
        config = settings.embedding
        provider = config.provider.lower()
        
        if provider == "openai":
            # B7.3 阶段实现
            raise NotImplementedError(
                "OpenAI Embedding 实现将在 B7.3 阶段完成。"
                "请先使用其他 provider 或等待实现。"
            )
        elif provider == "local":
            # B7.4 阶段实现
            raise NotImplementedError(
                "Local Embedding 实现将在 B7.4 阶段完成。"
                "请先使用其他 provider 或等待实现。"
            )
        elif provider == "ollama":
            # B7.4 阶段实现
            raise NotImplementedError(
                "Ollama Embedding 实现将在 B7.4 阶段完成。"
                "请先使用其他 provider 或等待实现。"
            )
        else:
            raise ValueError(
                f"不支持的 Embedding provider: {provider}。"
                f"支持的 provider: openai, local, ollama"
            )
    
    @staticmethod
    def create_fake(
        provider: str = "fake",
        model: str = "fake-model",
        dimension: int = 128
    ) -> BaseEmbedding:
        """
        创建 Fake Embedding 实例（用于测试）
        
        Args:
            provider: provider 名称
            model: 模型名称
            dimension: 向量维度
            
        Returns:
            BaseEmbedding: Fake Embedding 实例
        """
        from src.libs.embedding.fake_embedding import FakeEmbedding
        return FakeEmbedding(provider=provider, model=model, dimension=dimension)
