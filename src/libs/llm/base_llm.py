"""
LLM 抽象接口模块

定义统一的 LLM 接口，所有 LLM 实现（Azure、OpenAI、Ollama、DeepSeek 等）
都必须遵循此接口。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """
    LLM 抽象基类
    
    定义所有 LLM 实现必须遵循的统一接口。
    无论底层使用 Azure OpenAI、OpenAI、Ollama 还是 DeepSeek，
    上层代码都可以通过统一的接口调用。
    """
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        发送消息并获取 LLM 回复
        
        Args:
            messages: 消息列表，格式如：
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "What is Python?"}
                ]
                支持的 role: "system", "user", "assistant"
                
        Returns:
            str: LLM 的回复文本
            
        Raises:
            ValueError: 当消息格式不正确时
            RuntimeError: 当 LLM 调用失败时（网络错误、API 错误等）
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        获取当前使用的模型名称
        
        Returns:
            str: 模型名称，例如 "gpt-4o", "llama-3-8b"
        """
        pass
    
    @abstractmethod
    def get_provider(self) -> str:
        """
        获取当前使用的 provider 名称
        
        Returns:
            str: provider 名称，例如 "azure", "openai", "ollama", "deepseek"
        """
        pass
