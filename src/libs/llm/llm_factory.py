"""
LLM 工厂模块

根据配置创建对应的 LLM 实现实例。
支持通过配置文件切换不同的 LLM provider，无需修改代码。
"""
from typing import Optional

from src.core.settings import Settings
from src.libs.llm.base_llm import BaseLLM


class LLMFactory:
    """
    LLM 工厂类
    
    根据配置动态创建对应的 LLM 实现。
    使用工厂模式实现配置驱动的组件选择。
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseLLM:
        """
        根据配置创建 LLM 实例
        
        Args:
            settings: 配置对象，包含 LLM 配置信息
            
        Returns:
            BaseLLM: LLM 实例
            
        Raises:
            ValueError: 当 provider 不支持或配置不完整时
            NotImplementedError: 当 provider 的实现在 B7 阶段尚未完成时
        """
        config = settings.llm
        provider = config.provider.lower()
        
        if provider == "azure":
            # B7.1 阶段实现
            from src.libs.llm.azure_llm import AzureLLM
            return AzureLLM(config)
        elif provider == "openai":
            # B7.1 阶段实现
            from src.libs.llm.openai_llm import OpenAILLM
            return OpenAILLM(config)
        elif provider == "ollama":
            # B7.2 阶段实现
            raise NotImplementedError(
                "Ollama LLM 实现将在 B7.2 阶段完成。"
                "请先使用其他 provider 或等待实现。"
            )
        elif provider == "deepseek":
            # B7.1 阶段实现
            from src.libs.llm.deepseek_llm import DeepSeekLLM
            return DeepSeekLLM(config)
        else:
            raise ValueError(
                f"不支持的 LLM provider: {provider}。"
                f"支持的 provider: azure, openai, ollama, deepseek"
            )
    
    @staticmethod
    def create_fake(provider: str = "fake", model: str = "fake-model") -> BaseLLM:
        """
        创建 Fake LLM 实例（用于测试）
        
        Args:
            provider: provider 名称
            model: 模型名称
            
        Returns:
            BaseLLM: Fake LLM 实例
        """
        from src.libs.llm.fake_llm import FakeLLM
        return FakeLLM(provider=provider, model=model)
