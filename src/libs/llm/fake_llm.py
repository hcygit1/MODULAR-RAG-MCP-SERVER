"""
Fake LLM 实现（用于测试）

提供一个简单的 Fake LLM 实现，用于测试工厂路由逻辑，
不进行真实的 LLM API 调用。
"""
from typing import List, Dict

from src.libs.llm.base_llm import BaseLLM


class FakeLLM(BaseLLM):
    """
    Fake LLM 实现
    
    用于测试和开发阶段，返回固定的回复，不进行真实的 API 调用。
    """
    
    def __init__(self, provider: str = "fake", model: str = "fake-model"):
        """
        初始化 Fake LLM
        
        Args:
            provider: provider 名称
            model: 模型名称
        """
        self._provider = provider
        self._model = model
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        返回固定的 Fake 回复
        
        Args:
            messages: 消息列表
            
        Returns:
            str: Fake 回复文本
        """
        if not messages:
            raise ValueError("消息列表不能为空")
        
        # 提取最后一条用户消息
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
        
        if last_user_message:
            return f"[Fake LLM Response] You said: {last_user_message}"
        else:
            return "[Fake LLM Response] Hello! This is a fake LLM."
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self._model
    
    def get_provider(self) -> str:
        """获取 provider 名称"""
        return self._provider
