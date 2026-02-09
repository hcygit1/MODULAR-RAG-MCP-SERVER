"""
DashScope Embedding 实现

使用阿里云 DashScope API 的 Qwen Embedding 实现。
"""
import json
from typing import List, Optional, Any
import urllib.request
import urllib.error

from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.settings import EmbeddingConfig


class DashScopeEmbedding(BaseEmbedding):
    """
    DashScope Embedding 实现
    
    使用阿里云 DashScope API 进行文本向量化。
    支持批量处理，自动处理超长文本（截断或报错）。
    """
    
    # DashScope Embedding 模型的维度映射
    MODEL_DIMENSIONS = {
        "text-embedding-v1": 1536,
        "text-embedding-v2": 1536,
    }
    
    def __init__(self, config: EmbeddingConfig):
        """
        初始化 DashScope Embedding
        
        Args:
            config: Embedding 配置对象
        """
        if not config.dashscope_api_key:
            raise ValueError("DashScope API key 不能为空")
        
        if not config.model:
            raise ValueError("DashScope Embedding model 名称不能为空")
        
        self._config = config
        self._provider = config.provider.lower()  # 从配置中获取 provider
        self._model = config.model
        self._api_key = config.dashscope_api_key
        self._base_url = config.dashscope_base_url.rstrip("/")
        
        # 获取模型维度
        self._dimension = self.MODEL_DIMENSIONS.get(
            self._model,
            1536  # 默认维度
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
                f"DashScope Embedding API 调用失败 (provider={self._provider}, model={self._model}): {error_msg}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"DashScope Embedding API 网络错误 (provider={self._provider}, model={self._model}): {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"DashScope Embedding 调用失败 (provider={self._provider}, model={self._model}): {str(e)}"
            ) from e
    
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        调用 DashScope Embedding API
        
        DashScope Embedding API 格式：
        POST https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
        {
            "model": "text-embedding-v1",
            "input": {
                "texts": ["text1", "text2", ...]
            }
        }
        
        Args:
            texts: 文本列表
        
        Returns:
            List[List[float]]: 向量列表
        """
        url = f"{self._base_url}/api/v1/services/embeddings/text-embedding/text-embedding"
        
        payload = {
            "model": self._model,
            "input": {
                "texts": texts
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
            
            # DashScope Embedding API 响应格式：
            # {
            #     "output": {
            #         "embeddings": [
            #             {
            #                 "embedding": [0.1, 0.2, ...],
            #                 "text_index": 0
            #             },
            #             ...
            #         ]
            #     },
            #     "usage": {...},
            #     "request_id": "..."
            # }
            if "output" not in response_data:
                raise RuntimeError("API 响应格式错误：缺少 output 字段")
            
            output = response_data["output"]
            if "embeddings" not in output:
                raise RuntimeError("API 响应格式错误：缺少 embeddings 字段")
            
            # 按 text_index 排序，确保顺序正确
            embeddings_data = sorted(
                output["embeddings"],
                key=lambda x: x.get("text_index", 0)
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
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self._dimension
