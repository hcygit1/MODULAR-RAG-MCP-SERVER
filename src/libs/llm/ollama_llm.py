"""
Ollama LLM 实现

使用 Ollama 本地 API 的 LLM 实现。
Ollama 是一个本地运行的 LLM 服务，无需 API key。
"""
import json
from typing import List, Dict
import urllib.request
import urllib.error

from src.libs.llm.base_llm import BaseLLM
from src.core.settings import LLMConfig


class OllamaLLM(BaseLLM):
    """
    Ollama LLM 实现
    
    使用 Ollama 本地 API 进行对话。
    Ollama 是一个本地运行的 LLM 服务，默认运行在 http://localhost:11434
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化 Ollama LLM
        
        Args:
            config: LLM 配置对象
        """
        if not config.model:
            raise ValueError("Ollama model 名称不能为空")
        
        self._config = config
        self._provider = config.provider.lower()  # 从配置中获取 provider
        self._model = config.model
        self._base_url = config.ollama_base_url.rstrip("/")
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        发送消息并获取 Ollama 回复
        
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
            RuntimeError: 当 API 调用失败时（网络错误、连接失败、超时等）
        """
        self._validate_messages(messages)
        
        try:
            return self._call_api(messages)
        except urllib.error.HTTPError as e:
            error_msg = self._parse_error_response(e)
            raise RuntimeError(
                f"Ollama API 调用失败 (provider={self._provider}, model={self._model}): {error_msg}"
            ) from e
        except urllib.error.URLError as e:
            # 连接失败/超时等场景，不泄露 base_url 等敏感信息
            error_reason = str(e.reason) if hasattr(e, 'reason') else str(e)
            raise RuntimeError(
                f"Ollama API 连接失败 (provider={self._provider}, model={self._model}): "
                f"无法连接到 Ollama 服务，请确保 Ollama 正在运行"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Ollama LLM 调用失败 (provider={self._provider}, model={self._model}): {str(e)}"
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
        调用 Ollama API
        
        Ollama API 格式：
        POST {base_url}/api/chat
        {
            "model": "model-name",
            "messages": [...],
            "stream": false
        }
        
        Args:
            messages: 消息列表
        
        Returns:
            str: API 返回的回复文本
        """
        url = f"{self._base_url}/api/chat"
        
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False  # 非流式响应
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            
            # Ollama API 响应格式：
            # {
            #     "model": "model-name",
            #     "created_at": "...",
            #     "message": {
            #         "role": "assistant",
            #         "content": "回复内容"
            #     },
            #     "done": true
            # }
            if "message" not in response_data:
                raise RuntimeError("API 响应格式错误：缺少 message 字段")
            
            message = response_data["message"]
            if "content" not in message:
                raise RuntimeError("API 响应格式错误：缺少 message.content 字段")
            
            return message["content"]
    
    def _parse_error_response(self, error: urllib.error.HTTPError) -> str:
        """解析 HTTP 错误响应，不泄露敏感配置"""
        try:
            error_body = error.read().decode("utf-8")
            error_data = json.loads(error_body)
            
            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, str):
                    return error_info
                elif isinstance(error_info, dict) and "message" in error_info:
                    return error_info["message"]
                return str(error_info)
            
            return f"HTTP {error.code}: {error.reason}"
        except Exception:
            # 解析失败时，只返回状态码和原因，不泄露响应体内容
            return f"HTTP {error.code}: {error.reason}"
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self._model
    
    def get_provider(self) -> str:
        """获取 provider 名称"""
        return self._provider
