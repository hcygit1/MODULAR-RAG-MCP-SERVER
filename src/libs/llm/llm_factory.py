"""
LLM 工厂模块

根据配置创建对应的 LLM 实现实例。
支持通过配置文件切换不同的 LLM provider，无需修改代码。
支持普通 LLM 和 Vision LLM。
"""
from typing import Optional, Union

from src.core.settings import Settings, LLMConfig, VisionLLMConfig
from src.libs.llm.base_llm import BaseLLM


class LLMFactory:
    """
    LLM 工厂类
    
    根据配置动态创建对应的 LLM 实现。
    使用工厂模式实现配置驱动的组件选择。
    支持普通 LLM 和 Vision LLM。
    """
    
    @staticmethod
    def create(
        settings: Optional[Settings] = None,
        config: Optional[Union[LLMConfig, VisionLLMConfig]] = None
    ) -> BaseLLM:
        """
        根据配置创建 LLM 实例
        
        Args:
            settings: 配置对象（可选），如果提供则使用 settings.llm
            config: 配置对象（可选），可以是 LLMConfig 或 VisionLLMConfig
                   如果同时提供 settings 和 config，优先使用 config
            
        Returns:
            BaseLLM: LLM 实例
            
        Raises:
            ValueError: 当 provider 不支持或配置不完整时
            NotImplementedError: 当 provider 的实现在 B7 阶段尚未完成时
        """
        # 确定使用哪个配置
        if config is None:
            if settings is None:
                raise ValueError("必须提供 settings 或 config 参数")
            config = settings.llm
        
        provider = config.provider.lower()
        is_vision = isinstance(config, VisionLLMConfig)
        
        # 处理 Vision LLM 的特殊情况
        if is_vision and (provider == "dashscope" or provider == "qwen"):
            # DashScope/Qwen Vision LLM 实现
            from src.libs.llm.dashscope_vision_llm import DashScopeVisionLLM
            return DashScopeVisionLLM(config)
        
        # 对于需要 LLMConfig 的实现，如果传入的是 VisionLLMConfig，需要适配
        if is_vision and provider == "azure":
            # Azure Vision LLM：将 VisionLLMConfig 适配为 LLMConfig
            llm_config = LLMConfig(
                provider="azure",
                model=config.model,
                azure_endpoint=config.azure_endpoint,
                azure_api_key=config.azure_api_key,
                azure_api_version=config.azure_api_version,
                deployment_name=config.deployment_name
            )
            from src.libs.llm.azure_llm import AzureLLM
            return AzureLLM(llm_config)
        
        # 普通 LLM 实现（需要 LLMConfig）
        if is_vision:
            raise ValueError(
                f"Vision LLM provider '{provider}' 暂不支持，"
                f"支持的 Vision LLM provider: azure, dashscope, qwen"
            )
        
        # 确保是 LLMConfig
        if not isinstance(config, LLMConfig):
            raise ValueError(f"普通 LLM 需要 LLMConfig，得到: {type(config)}")
        
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
            from src.libs.llm.ollama_llm import OllamaLLM
            return OllamaLLM(config)
        elif provider == "deepseek":
            # B7.1 阶段实现
            from src.libs.llm.deepseek_llm import DeepSeekLLM
            return DeepSeekLLM(config)
        elif provider == "dashscope" or provider == "qwen":
            # Qwen LLM 实现
            from src.libs.llm.qwen_llm import QwenLLM
            return QwenLLM(config)
        else:
            raise ValueError(
                f"不支持的 LLM provider: {provider}。"
                f"支持的 provider: azure, openai, ollama, deepseek, dashscope, qwen"
            )
    
    @staticmethod
    def create_vision_llm(settings: Settings) -> Optional[BaseLLM]:
        """
        根据配置创建 Vision LLM 实例（便捷方法）
        
        Args:
            settings: 配置对象，包含 Vision LLM 配置信息
            
        Returns:
            Optional[BaseLLM]: Vision LLM 实例，如果配置禁用或不可用则返回 None
        """
        try:
            return LLMFactory.create(config=settings.vision_llm)
        except ValueError:
            # 不支持的 provider，返回 None（走降级路径）
            return None
    
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
