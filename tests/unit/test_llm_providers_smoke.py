"""
LLM Providers 冒烟测试

使用 mock HTTP 测试各个 LLM provider 的实现，不走真实网络。
"""
import json
from unittest.mock import patch, MagicMock
import pytest
import urllib.error

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM


def test_factory_creates_openai_llm():
    """测试工厂可以创建 OpenAI LLM"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    llm = LLMFactory.create(settings)
    
    assert isinstance(llm, OpenAILLM)
    assert llm.get_provider() == "openai"
    assert llm.get_model_name() == "gpt-4o"


def test_factory_creates_azure_llm():
    """测试工厂可以创建 Azure LLM"""
    llm_config = LLMConfig(
        provider="azure",
        model="gpt-4o",
        azure_endpoint="https://test.openai.azure.com",
        azure_api_key="test-key",
        deployment_name="test-deployment"
    )
    settings = _create_test_settings(llm_config)
    
    llm = LLMFactory.create(settings)
    
    assert isinstance(llm, AzureLLM)
    assert llm.get_provider() == "azure"
    assert llm.get_model_name() == "gpt-4o"


def test_factory_creates_deepseek_llm():
    """测试工厂可以创建 DeepSeek LLM"""
    llm_config = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        deepseek_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    llm = LLMFactory.create(settings)
    
    assert isinstance(llm, DeepSeekLLM)
    assert llm.get_provider() == "deepseek"
    assert llm.get_model_name() == "deepseek-chat"


@patch("urllib.request.urlopen")
def test_openai_llm_chat_success(mock_urlopen):
    """测试 OpenAI LLM chat 成功调用"""
    # Mock HTTP 响应
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": "Hello! How can I help you?"
            }
        }]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = OpenAILLM(LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    ))
    
    messages = [{"role": "user", "content": "Hello"}]
    response = llm.chat(messages)
    
    assert response == "Hello! How can I help you?"
    assert mock_urlopen.called


@patch("urllib.request.urlopen")
def test_azure_llm_chat_success(mock_urlopen):
    """测试 Azure LLM chat 成功调用"""
    # Mock HTTP 响应
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": "Azure response"
            }
        }]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = AzureLLM(LLMConfig(
        provider="azure",
        model="gpt-4o",
        azure_endpoint="https://test.openai.azure.com",
        azure_api_key="test-key",
        deployment_name="test-deployment"
    ))
    
    messages = [{"role": "user", "content": "Hello"}]
    response = llm.chat(messages)
    
    assert response == "Azure response"
    assert mock_urlopen.called


@patch("urllib.request.urlopen")
def test_deepseek_llm_chat_success(mock_urlopen):
    """测试 DeepSeek LLM chat 成功调用"""
    # Mock HTTP 响应
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": "DeepSeek response"
            }
        }]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = DeepSeekLLM(LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        deepseek_api_key="test-key"
    ))
    
    messages = [{"role": "user", "content": "Hello"}]
    response = llm.chat(messages)
    
    assert response == "DeepSeek response"
    assert mock_urlopen.called


def test_openai_llm_missing_api_key():
    """测试 OpenAI LLM 缺少 API key 时抛出错误"""
    with pytest.raises(ValueError, match="OpenAI API key 不能为空"):
        OpenAILLM(LLMConfig(
            provider="openai",
            model="gpt-4o",
            openai_api_key=""
        ))


def test_azure_llm_missing_endpoint():
    """测试 Azure LLM 缺少 endpoint 时抛出错误"""
    with pytest.raises(ValueError, match="Azure endpoint 不能为空"):
        AzureLLM(LLMConfig(
            provider="azure",
            model="gpt-4o",
            azure_endpoint="",
            azure_api_key="test-key",
            deployment_name="test"
        ))


def test_azure_llm_missing_api_key():
    """测试 Azure LLM 缺少 API key 时抛出错误"""
    with pytest.raises(ValueError, match="Azure API key 不能为空"):
        AzureLLM(LLMConfig(
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            azure_api_key="",
            deployment_name="test"
        ))


def test_azure_llm_missing_deployment_name():
    """测试 Azure LLM 缺少 deployment name 时抛出错误"""
    with pytest.raises(ValueError, match="Azure deployment name 不能为空"):
        AzureLLM(LLMConfig(
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            azure_api_key="test-key",
            deployment_name=""
        ))


def test_deepseek_llm_missing_api_key():
    """测试 DeepSeek LLM 缺少 API key 时抛出错误"""
    with pytest.raises(ValueError, match="DeepSeek API key 不能为空"):
        DeepSeekLLM(LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            deepseek_api_key=""
        ))


def test_openai_llm_empty_messages():
    """测试 OpenAI LLM 处理空消息列表"""
    llm = OpenAILLM(LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(ValueError, match="消息列表不能为空"):
        llm.chat([])


def test_openai_llm_invalid_message_format():
    """测试 OpenAI LLM 处理无效消息格式"""
    llm = OpenAILLM(LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    ))
    
    # 缺少 role
    with pytest.raises(ValueError, match="缺少 'role' 字段"):
        llm.chat([{"content": "test"}])
    
    # 缺少 content
    with pytest.raises(ValueError, match="缺少 'content' 字段"):
        llm.chat([{"role": "user"}])
    
    # 无效的 role
    with pytest.raises(ValueError, match="role 必须是"):
        llm.chat([{"role": "invalid", "content": "test"}])


@patch("urllib.request.urlopen")
def test_openai_llm_http_error(mock_urlopen):
    """测试 OpenAI LLM 处理 HTTP 错误"""
    # Mock HTTP 错误
    error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=None
    )
    error.read = MagicMock(return_value=json.dumps({
        "error": {"message": "Invalid API key"}
    }).encode("utf-8"))
    mock_urlopen.side_effect = error
    
    llm = OpenAILLM(LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "Hello"}])
    
    assert "OpenAI API 调用失败" in str(exc_info.value)
    assert "provider=openai" in str(exc_info.value)
    assert "model=gpt-4o" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_azure_llm_http_error(mock_urlopen):
    """测试 Azure LLM 处理 HTTP 错误"""
    # Mock HTTP 错误
    error = urllib.error.HTTPError(
        url="https://test.openai.azure.com/...",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=None
    )
    error.read = MagicMock(return_value=json.dumps({
        "error": {"message": "Invalid request"}
    }).encode("utf-8"))
    mock_urlopen.side_effect = error
    
    llm = AzureLLM(LLMConfig(
        provider="azure",
        model="gpt-4o",
        azure_endpoint="https://test.openai.azure.com",
        azure_api_key="test-key",
        deployment_name="test-deployment"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "Hello"}])
    
    assert "Azure OpenAI API 调用失败" in str(exc_info.value)
    assert "provider=azure" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_deepseek_llm_http_error(mock_urlopen):
    """测试 DeepSeek LLM 处理 HTTP 错误"""
    # Mock HTTP 错误
    error = urllib.error.HTTPError(
        url="https://api.deepseek.com/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=None
    )
    error.read = MagicMock(return_value=json.dumps({
        "error": {"message": "Rate limit exceeded"}
    }).encode("utf-8"))
    mock_urlopen.side_effect = error
    
    llm = DeepSeekLLM(LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        deepseek_api_key="test-key"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "Hello"}])
    
    assert "DeepSeek API 调用失败" in str(exc_info.value)
    assert "provider=deepseek" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_openai_llm_network_error(mock_urlopen):
    """测试 OpenAI LLM 处理网络错误"""
    # Mock 网络错误
    mock_urlopen.side_effect = urllib.error.URLError("Connection timeout")
    
    llm = OpenAILLM(LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "Hello"}])
    
    assert "OpenAI API 网络错误" in str(exc_info.value)
    assert "provider=openai" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_openai_llm_invalid_response_format(mock_urlopen):
    """测试 OpenAI LLM 处理无效响应格式"""
    # Mock 无效响应
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "invalid": "response"
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = OpenAILLM(LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    ))
    
    with pytest.raises(RuntimeError, match="API 响应格式错误"):
        llm.chat([{"role": "user", "content": "Hello"}])


def test_factory_provider_routing():
    """测试工厂路由不同 provider"""
    # 测试 openai
    llm_config_openai = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings_openai = _create_test_settings(llm_config_openai)
    llm_openai = LLMFactory.create(settings_openai)
    assert isinstance(llm_openai, OpenAILLM)
    
    # 测试 azure
    llm_config_azure = LLMConfig(
        provider="azure",
        model="gpt-4o",
        azure_endpoint="https://test.openai.azure.com",
        azure_api_key="test-key",
        deployment_name="test-deployment"
    )
    settings_azure = _create_test_settings(llm_config_azure)
    llm_azure = LLMFactory.create(settings_azure)
    assert isinstance(llm_azure, AzureLLM)
    
    # 测试 deepseek
    llm_config_deepseek = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        deepseek_api_key="test-key"
    )
    settings_deepseek = _create_test_settings(llm_config_deepseek)
    llm_deepseek = LLMFactory.create(settings_deepseek)
    assert isinstance(llm_deepseek, DeepSeekLLM)


def _create_test_settings(llm_config: LLMConfig) -> Settings:
    """创建测试用的 Settings 对象"""
    return Settings(
        llm=llm_config,
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
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
