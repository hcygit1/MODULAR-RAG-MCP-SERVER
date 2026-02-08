"""
OpenAI Embedding 实现

使用 OpenAI Embedding API 的实现。
"""
import json
from typing import List, Optional, Any
import urllib.request
import urllib.error

from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.settings import EmbeddingConfig


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embedding 实现
    
    使用 OpenAI Embedding API 进行文本向量化。
    支持批量处理，自动处理超长文本（截断或报错）。
    """
    
    # OpenAI Embedding 模型的维度映射
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, config: EmbeddingConfig):
        """
        初始化 OpenAI Embedding
        
        Args:
            config: Embedding 配置对象
        """
        if not config.openai_api_key:
            raise ValueError("OpenAI API key 不能为空")
        
        if not config.model:
            raise ValueError("OpenAI Embedding model 名称不能为空")
        
        self._config = config
        self._provider = "openai"
        self._model = config.model
        self._api_key = config.openai_api_key
        self._base_url = "https://api.openai.com/v1"
        
        # 获取模型维度
        self._dimension = self.MODEL_DIMENSIONS.get(
            self._model,
            1536  # 默认维度（text-embedding-3-small）
        )
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None
    ) -> List[List[float]]:
        """
        批量将文本转换为向量
        
        Args:
            texts: 文本列表，每个元素是一个字符串
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
        
        Returns:
            List[List[float]]: 向量列表，每个文本对应一个向量
        
        Raises:
            ValueError: 当文本列表为空或包含无效文本时
            RuntimeError: 当 API 调用失败时（网络错误、API 错误等）
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        # 验证文本格式
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"文本 {i} 必须是字符串类型，得到: {type(text)}")
            if not text.strip():
                raise ValueError(f"文本 {i} 不能为空")
        
        try:
            return self._call_api(texts)
        except urllib.error.HTTPError as e:
            error_msg = self._parse_error_response(e)
            raise RuntimeError(
                f"OpenAI Embedding API 调用失败 (provider={self._provider}, model={self._model}): {error_msg}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"OpenAI Embedding API 网络错误 (provider={self._provider}, model={self._model}): {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"OpenAI Embedding 调用失败 (provider={self._provider}, model={self._model}): {str(e)}"
            ) from e
    
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        调用 OpenAI Embedding API
        
        OpenAI Embedding API 格式：
        POST https://api.openai.com/v1/embeddings
        {
            "model": "text-embedding-3-small",
            "input": ["text1", "text2", ...]
        }
        
        Args:
            texts: 文本列表
        
        Returns:
            List[List[float]]: 向量列表
        """
        url = f"{self._base_url}/embeddings"
        
        payload = {
            "model": self._model,
            "input": texts
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
            
            # OpenAI Embedding API 响应格式：
            # {
            #     "object": "list",
            #     "data": [
            #         {
            #             "object": "embedding",
            #             "embedding": [0.1, 0.2, ...],
            #             "index": 0
            #         },
            #         ...
            #     ],
            #     "model": "text-embedding-3-small",
            #     "usage": {...}
            # }
            if "data" not in response_data:
                raise RuntimeError("API 响应格式错误：缺少 data 字段")
            
            # 按 index 排序，确保顺序正确
            embeddings_data = sorted(
                response_data["data"],
                key=lambda x: x.get("index", 0)
            )
            
            vectors = []
            for item in embeddings_data:
                if "embedding" not in item:
                    raise RuntimeError("API 响应格式错误：缺少 embedding 字段")
                vectors.append(item["embedding"])
            
            return vectors
    
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
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self._dimension
