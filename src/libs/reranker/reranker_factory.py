"""
Reranker 工厂模块

根据配置创建对应的 Reranker 实现实例。
支持通过配置文件切换不同的重排序策略，无需修改代码。
"""
from typing import Optional

from src.core.settings import Settings
from src.libs.reranker.base_reranker import BaseReranker


class RerankerFactory:
    """
    Reranker 工厂类
    
    根据配置动态创建对应的 Reranker 实现。
    使用工厂模式实现配置驱动的组件选择。
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseReranker:
        """
        根据配置创建 Reranker 实例
        
        Args:
            settings: 配置对象，包含 Rerank 配置信息
        
        Returns:
            BaseReranker: Reranker 实例
        
        Raises:
            ValueError: 当 backend 不支持或配置不完整时
            NotImplementedError: 当 backend 的实现在 B7 阶段尚未完成时
        """
        config = settings.rerank
        backend = config.backend.lower()
        
        if backend == "none" or backend == "":
            # B5 阶段实现
            from src.libs.reranker.none_reranker import NoneReranker
            return NoneReranker()
        elif backend == "cross_encoder":
            # B7.8 阶段实现
            from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
            return CrossEncoderReranker(config)
        elif backend == "llm":
            # B7.7 阶段实现
            from src.libs.reranker.llm_reranker import LLMReranker
            return LLMReranker(settings)
        else:
            raise ValueError(
                f"不支持的 Reranker backend: {backend}。"
                f"支持的 backend: none, cross_encoder, llm"
            )
    
    @staticmethod
    def create_none() -> BaseReranker:
        """
        创建 NoneReranker 实例（用于测试）
        
        Returns:
            BaseReranker: NoneReranker 实例
        """
        from src.libs.reranker.none_reranker import NoneReranker
        return NoneReranker()
