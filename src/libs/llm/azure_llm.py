"""
Azure OpenAI LLM 实现

使用 Azure OpenAI API 的 LLM 实现。
"""
import json
from typing import List, Dict
import urllib.request
import urllib.error

from src.libs.llm.base_llm import BaseLLM
from src.core.settings import LLMConfig


class AzureLLM(BaseLLM):
    """
    Azure OpenAI LLM 实现
    
    使用 Azure OpenAI API 进行对话。
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化 Azure OpenAI LLM
        
        Args:
            config: LLM 配置对象
        """
        if not config.azure_endpoint:
            raise ValueError("Azure endpoint 不能为空")
        
        if not config.azure_api_key:
            raise ValueError("Azure API key 不能为空")
        
        if not config.deployment_name:
            raise ValueError("Azure deployment name 不能为空")
        
        self._config = config
        self._provider = config.provider.lower()  # 从配置中获取 provider
        self._model = config.model
        self._endpoint = config.azure_endpoint.rstrip("/")
        self._api_key = config.azure_api_key
        self._api_version = config.azure_api_version
        self._deployment_name = config.deployment_name
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        发送消息并获取 Azure OpenAI 回复
        
        Args:
            messages: 消息列表，格式如：
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
        
        Returns:
            str: LLM 的回复文本
        
        Raises:
            ValueError: 当消息格式不正确时
            RuntimeError: 当 API 调用失败时（网络错误、API 错误等）
        """
        self._validate_messages(messages)
        
        try:
            return self._call_api(messages)
        except urllib.error.HTTPError as e:
            error_msg = self._parse_error_response(e)
            raise RuntimeError(
                f"Azure OpenAI API 调用失败 (provider={self._provider}, model={self._model}): {error_msg}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Azure OpenAI API 网络错误 (provider={self._provider}, model={self._model}): {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Azure LLM 调用失败 (provider={self._provider}, model={self._model}): {str(e)}"
            ) from e
    
    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """验证消息格式"""
        if not messages:
            raise ValueError("消息列表不能为空")
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"消息 {i} 必须是字典类型，得到: {type(msg)}")
            
            if "role" not in msg:
                raise ValueError(f"消息 {i} 缺少 'role' 字段")
            
            if "content" not in msg:
                raise ValueError(f"消息 {i} 缺少 'content' 字段")
            
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    f"消息 {i} 的 role 必须是 'system', 'user' 或 'assistant'，"
                    f"得到: {msg['role']}"
                )
    
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        调用 Azure OpenAI API
        
        Args:
            messages: 消息列表
        
        Returns:
            str: API 返回的回复文本
        """
        # Azure OpenAI API 格式：
        # {endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}
        url = (
            f"{self._endpoint}/openai/deployments/{self._deployment_name}"
            f"/chat/completions?api-version={self._api_version}"
        )
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            
            # 提取回复文本
            if "choices" not in response_data or not response_data["choices"]:
                raise RuntimeError("API 响应格式错误：缺少 choices 字段")
            
            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                raise RuntimeError("API 响应格式错误：缺少 message.content 字段")
            
            return choice["message"]["content"]
    
    def _parse_error_response(self, error: urllib.error.HTTPError) -> str:
        """解析 HTTP 错误响应"""
        try:
            error_body = error.read().decode("utf-8")
            error_data = json.loads(error_body)
            
            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict) and "message" in error_info:
                    return error_info["message"]
                return str(error_info)
            
            return f"HTTP {error.code}: {error.reason}"
        except Exception:
            return f"HTTP {error.code}: {error.reason}"
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self._model
    
    def get_provider(self) -> str:
        """获取 provider 名称"""
        return self._provider
