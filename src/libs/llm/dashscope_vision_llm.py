"""
DashScope Vision LLM 实现

使用阿里云 DashScope API 的 Qwen-VL Vision LLM 实现。
支持图片输入和文本描述生成。
"""
import json
import base64
from typing import List, Dict, Optional
from pathlib import Path
import urllib.request
import urllib.error

from src.libs.llm.base_llm import BaseLLM
from src.core.settings import VisionLLMConfig


class DashScopeVisionLLM(BaseLLM):
    """
    DashScope Vision LLM 实现
    
    使用阿里云 DashScope API 的 Qwen-VL 模型进行多模态对话。
    支持图片输入和文本描述生成。
    """
    
    def __init__(self, config: VisionLLMConfig):
        """
        初始化 DashScope Vision LLM
        
        Args:
            config: Vision LLM 配置对象
        """
        if not config.dashscope_api_key:
            raise ValueError("DashScope API key 不能为空")
        
        self._config = config
        self._provider = config.provider.lower()  # 从配置中获取 provider
        self._model = config.model
        self._api_key = config.dashscope_api_key
        self._base_url = config.dashscope_base_url.rstrip("/")
    
    def chat(self, messages: List[Dict[str, str]], image_path: Optional[str] = None) -> str:
        """
        发送消息并获取 Vision LLM 回复
        
        Args:
            messages: 消息列表，格式如：
                [
                    {"role": "user", "content": "描述这张图片"}
                ]
            image_path: 图片路径（可选），如果提供则会将图片编码后加入消息
        
        Returns:
            str: Vision LLM 的回复文本
        
        Raises:
            ValueError: 当消息格式不正确时
            RuntimeError: 当 API 调用失败时（网络错误、API 错误等）
        """
        self._validate_messages(messages)
        
        # 如果提供了图片路径，将图片编码并添加到消息中
        if image_path:
            messages = self._add_image_to_messages(messages, image_path)
        
        try:
            return self._call_api(messages)
        except urllib.error.HTTPError as e:
            error_msg = self._parse_error_response(e)
            raise RuntimeError(
                f"DashScope Vision API 调用失败 (provider={self._provider}, model={self._model}): {error_msg}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"DashScope Vision API 网络错误 (provider={self._provider}, model={self._model}): {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"DashScope Vision LLM 调用失败 (provider={self._provider}, model={self._model}): {str(e)}"
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
    
    def _add_image_to_messages(
        self,
        messages: List[Dict[str, str]],
        image_path: str
    ) -> List[Dict[str, str]]:
        """
        将图片编码并添加到消息中
        
        DashScope Qwen-VL 支持图片输入，格式：
        {
            "role": "user",
            "content": [
                {
                    "image": "base64_encoded_image"
                },
                {
                    "text": "描述这张图片"
                }
            ]
        }
        
        Args:
            messages: 原始消息列表
            image_path: 图片路径
        
        Returns:
            List[Dict]: 添加图片后的消息列表
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 读取图片并编码为 base64
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # 复制消息列表
        new_messages = messages.copy()
        
        # 找到最后一个 user 消息，添加图片
        if new_messages and new_messages[-1]["role"] == "user":
            # 如果 content 是字符串，转换为列表格式
            if isinstance(new_messages[-1]["content"], str):
                text_content = new_messages[-1]["content"]
                new_messages[-1]["content"] = [
                    {
                        "image": image_base64
                    },
                    {
                        "text": text_content
                    }
                ]
            elif isinstance(new_messages[-1]["content"], list):
                # 如果已经是列表，在开头添加图片
                new_messages[-1]["content"].insert(0, {"image": image_base64})
        else:
            # 如果没有 user 消息，创建一个新的
            new_messages.append({
                "role": "user",
                "content": [
                    {
                        "image": image_base64
                    },
                    {
                        "text": "请描述这张图片"
                    }
                ]
            })
        
        return new_messages
    
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        调用 DashScope Vision API
        
        DashScope Qwen-VL API 格式：
        POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
        {
            "model": "qwen-vl-max",
            "input": {
                "messages": [...]
            },
            "parameters": {
                "temperature": 0.7
            }
        }
        
        Args:
            messages: 消息列表（可能包含图片）
        
        Returns:
            str: API 返回的回复文本
        """
        url = f"{self._base_url}/api/v1/services/aigc/multimodal-generation/generation"
        
        payload = {
            "model": self._model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": 0.7
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            
            # DashScope Vision API 响应格式：
            # {
            #     "output": {
            #         "choices": [
            #             {
            #                 "message": {
            #                     "role": "assistant",
            #                     "content": "..."
            #                 }
            #             }
            #         ]
            #     },
            #     "usage": {...},
            #     "request_id": "..."
            # }
            if "output" not in response_data:
                raise RuntimeError("API 响应格式错误：缺少 output 字段")
            
            output = response_data["output"]
            if "choices" not in output or not output["choices"]:
                raise RuntimeError("API 响应格式错误：缺少 choices 字段")
            
            choice = output["choices"][0]
            if "message" not in choice:
                raise RuntimeError("API 响应格式错误：缺少 message 字段")
            
            message = choice["message"]
            # content 可能是字符串或列表
            if isinstance(message["content"], str):
                return message["content"]
            elif isinstance(message["content"], list):
                # 如果是列表，提取文本部分
                text_parts = [
                    item.get("text", "") for item in message["content"]
                    if isinstance(item, dict) and "text" in item
                ]
                return " ".join(text_parts)
            else:
                raise RuntimeError("API 响应格式错误：content 格式不正确")
    
    def _parse_error_response(self, error: urllib.error.HTTPError) -> str:
        """解析 HTTP 错误响应"""
        try:
            error_body = error.read().decode("utf-8")
            error_data = json.loads(error_body)
            
            if "message" in error_data:
                return error_data["message"]
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
