"""
Embedding Providers 冒烟测试

使用 mock HTTP 测试各个 Embedding provider 的实现，不走真实网络。
"""
import json
from unittest.mock import patch, MagicMock
import pytest
import urllib.error

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.openai_embedding import OpenAIEmbedding


def test_factory_creates_openai_embedding():
    """测试工厂可以创建 OpenAI Embedding"""
    embedding_config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(embedding_config)
    
    embedding = EmbeddingFactory.create(settings)
    
    assert isinstance(embedding, OpenAIEmbedding)
    assert embedding.get_provider() == "openai"
    assert embedding.get_model_name() == "text-embedding-3-small"
    assert embedding.get_dimension() == 1536


@patch("urllib.request.urlopen")
def test_openai_embedding_success(mock_urlopen):
    """测试 OpenAI Embedding 成功调用"""
    # Mock HTTP 响应
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3] * 512,  # 1536 维
                "index": 0
            },
            {
                "object": "embedding",
                "embedding": [0.4, 0.5, 0.6] * 512,  # 1536 维
                "index": 1
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    texts = ["Hello", "World"]
    vectors = embedding.embed(texts)
    
    assert len(vectors) == 2
    assert len(vectors[0]) == 1536
    assert len(vectors[1]) == 1536
    assert mock_urlopen.called


def test_openai_embedding_missing_api_key():
    """测试 OpenAI Embedding 缺少 API key 时抛出错误"""
    with pytest.raises(ValueError, match="OpenAI API key 不能为空"):
        OpenAIEmbedding(EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            openai_api_key=""
        ))


def test_openai_embedding_missing_model():
    """测试 OpenAI Embedding 缺少 model 时抛出错误"""
    with pytest.raises(ValueError, match="OpenAI Embedding model 名称不能为空"):
        OpenAIEmbedding(EmbeddingConfig(
            provider="openai",
            model="",
            openai_api_key="test-key"
        ))


def test_openai_embedding_empty_texts():
    """测试 OpenAI Embedding 处理空文本列表"""
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(ValueError, match="文本列表不能为空"):
        embedding.embed([])


def test_openai_embedding_empty_string():
    """测试 OpenAI Embedding 处理空字符串"""
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(ValueError, match="文本 0 不能为空"):
        embedding.embed([""])
    
    with pytest.raises(ValueError, match="文本 0 不能为空"):
        embedding.embed(["   "])  # 只有空白字符


def test_openai_embedding_invalid_text_type():
    """测试 OpenAI Embedding 处理无效文本类型"""
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(ValueError, match="文本 0 必须是字符串类型"):
        embedding.embed([123])  # 非字符串
    
    with pytest.raises(ValueError, match="文本 1 必须是字符串类型"):
        embedding.embed(["Hello", None])


@patch("urllib.request.urlopen")
def test_openai_embedding_http_error(mock_urlopen):
    """测试 OpenAI Embedding 处理 HTTP 错误"""
    # Mock HTTP 错误
    error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/embeddings",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=None
    )
    error.read = MagicMock(return_value=json.dumps({
        "error": {"message": "Invalid API key"}
    }).encode("utf-8"))
    mock_urlopen.side_effect = error
    
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        embedding.embed(["Hello"])
    
    assert "OpenAI Embedding API 调用失败" in str(exc_info.value)
    assert "provider=openai" in str(exc_info.value)
    assert "model=text-embedding-3-small" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_openai_embedding_network_error(mock_urlopen):
    """测试 OpenAI Embedding 处理网络错误"""
    # Mock 网络错误
    mock_urlopen.side_effect = urllib.error.URLError("Connection timeout")
    
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        embedding.embed(["Hello"])
    
    assert "OpenAI Embedding API 网络错误" in str(exc_info.value)
    assert "provider=openai" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_openai_embedding_invalid_response_format(mock_urlopen):
    """测试 OpenAI Embedding 处理无效响应格式"""
    # Mock 无效响应（缺少 data 字段）
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "object": "list"
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(RuntimeError, match="API 响应格式错误.*data"):
        embedding.embed(["Hello"])


@patch("urllib.request.urlopen")
def test_openai_embedding_missing_embedding_field(mock_urlopen):
    """测试 OpenAI Embedding 处理响应中缺少 embedding 字段"""
    # Mock 响应缺少 embedding 字段
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "data": [
            {
                "object": "embedding",
                "index": 0
            }
        ]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(RuntimeError, match="API 响应格式错误.*embedding"):
        embedding.embed(["Hello"])


@patch("urllib.request.urlopen")
def test_openai_embedding_batch_processing(mock_urlopen):
    """测试 OpenAI Embedding 批量处理"""
    # Mock HTTP 响应（多个文本）
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1] * 1536,
                "index": 0
            },
            {
                "object": "embedding",
                "embedding": [0.2] * 1536,
                "index": 1
            },
            {
                "object": "embedding",
                "embedding": [0.3] * 1536,
                "index": 2
            }
        ],
        "model": "text-embedding-3-small"
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    texts = ["Text 1", "Text 2", "Text 3"]
    vectors = embedding.embed(texts)
    
    assert len(vectors) == 3
    assert all(len(v) == 1536 for v in vectors)


@patch("urllib.request.urlopen")
def test_openai_embedding_response_order(mock_urlopen):
    """测试 OpenAI Embedding 响应顺序正确（按 index 排序）"""
    # Mock HTTP 响应（index 乱序）
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.3] * 1536,
                "index": 2
            },
            {
                "object": "embedding",
                "embedding": [0.1] * 1536,
                "index": 0
            },
            {
                "object": "embedding",
                "embedding": [0.2] * 1536,
                "index": 1
            }
        ],
        "model": "text-embedding-3-small"
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    texts = ["Text 1", "Text 2", "Text 3"]
    vectors = embedding.embed(texts)
    
    # 验证顺序正确（按 index 排序）
    assert len(vectors) == 3
    assert vectors[0][0] == 0.1  # index 0
    assert vectors[1][0] == 0.2  # index 1
    assert vectors[2][0] == 0.3  # index 2


def test_openai_embedding_model_dimensions():
    """测试 OpenAI Embedding 模型维度映射"""
    # text-embedding-3-small
    embedding1 = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    assert embedding1.get_dimension() == 1536
    
    # text-embedding-3-large
    embedding2 = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-large",
        openai_api_key="test-key"
    ))
    assert embedding2.get_dimension() == 3072
    
    # text-embedding-ada-002
    embedding3 = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-ada-002",
        openai_api_key="test-key"
    ))
    assert embedding3.get_dimension() == 1536
    
    # 未知模型（使用默认维度）
    embedding4 = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="unknown-model",
        openai_api_key="test-key"
    ))
    assert embedding4.get_dimension() == 1536  # 默认维度


def test_openai_embedding_trace_parameter():
    """测试 OpenAI Embedding 接受 trace 参数（即使暂未使用）"""
    embedding = OpenAIEmbedding(EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    ))
    
    # trace 参数应该被接受（即使为 None）
    # 这里只测试接口兼容性，实际功能在 F1 阶段实现
    assert hasattr(embedding, 'embed')
    # 验证方法签名包含 trace 参数
    import inspect
    sig = inspect.signature(embedding.embed)
    assert 'trace' in sig.parameters


def _create_test_settings(embedding_config: EmbeddingConfig) -> Settings:
    """创建测试用的 Settings 对象"""
    return Settings(
        llm=LLMConfig(provider="fake", model="fake-model"),
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        embedding=embedding_config,
        vector_store=VectorStoreConfig(
            backend="chroma",
            persist_path="./data/db/chroma",
            collection_name="test"
        ),
        retrieval=RetrievalConfig(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10
        ),
        rerank=RerankConfig(backend="none", model="", top_m=30, timeout_seconds=5),
        evaluation=EvaluationConfig(
            backends=["custom"], golden_test_set="./tests/fixtures/golden_test_set.json"
        ),
        observability=ObservabilityConfig(
            enabled=True,
            logging=LoggingConfig(log_file="./logs/traces.jsonl", log_level="INFO"),
            detail_level="standard",
            dashboard=DashboardConfig(enabled=True, port=8501)
        ),
        ingestion=IngestionConfig(
            chunk_size=512,
            chunk_overlap=50,
            enable_llm_refinement=False,
            enable_metadata_enrichment=True,
            enable_image_captioning=True,
            batch_size=32
        )
    )
