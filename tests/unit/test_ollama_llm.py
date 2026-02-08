"""
Ollama LLM 测试

使用 mock HTTP 测试 Ollama LLM 实现，不走真实网络。
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
from src.libs.llm.ollama_llm import OllamaLLM


def test_factory_creates_ollama_llm():
    """测试工厂可以创建 Ollama LLM"""
    llm_config = LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    )
    settings = _create_test_settings(llm_config)
    
    llm = LLMFactory.create(settings)
    
    assert isinstance(llm, OllamaLLM)
    assert llm.get_provider() == "ollama"
    assert llm.get_model_name() == "llama3:8b"


@patch("urllib.request.urlopen")
def test_ollama_llm_chat_success(mock_urlopen):
    """测试 Ollama LLM chat 成功调用"""
    # Mock HTTP 响应（Ollama API 格式）
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "model": "llama3:8b",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you?"
        },
        "done": True
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    messages = [{"role": "user", "content": "Hello"}]
    response = llm.chat(messages)
    
    assert response == "Hello! How can I help you?"
    assert mock_urlopen.called


def test_ollama_llm_missing_model():
    """测试 Ollama LLM 缺少 model 时抛出错误"""
    with pytest.raises(ValueError, match="Ollama model 名称不能为空"):
        OllamaLLM(LLMConfig(
            provider="ollama",
            model="",
            ollama_base_url="http://localhost:11434"
        ))


def test_ollama_llm_empty_messages():
    """测试 Ollama LLM 处理空消息列表"""
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    with pytest.raises(ValueError, match="消息列表不能为空"):
        llm.chat([])


def test_ollama_llm_invalid_message_format():
    """测试 Ollama LLM 处理无效消息格式"""
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
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
def test_ollama_llm_http_error(mock_urlopen):
    """测试 Ollama LLM 处理 HTTP 错误"""
    # Mock HTTP 错误
    error = urllib.error.HTTPError(
        url="http://localhost:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs=None,
        fp=None
    )
    error.read = MagicMock(return_value=json.dumps({
        "error": "model 'llama3:8b' not found"
    }).encode("utf-8"))
    mock_urlopen.side_effect = error
    
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "Hello"}])
    
    assert "Ollama API 调用失败" in str(exc_info.value)
    assert "provider=ollama" in str(exc_info.value)
    assert "model=llama3:8b" in str(exc_info.value)
    # 确保不泄露 base_url 等敏感信息
    assert "localhost:11434" not in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_ollama_llm_connection_error(mock_urlopen):
    """测试 Ollama LLM 处理连接失败（不泄露敏感配置）"""
    # Mock 连接错误（URLError）
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "Hello"}])
    
    assert "Ollama API 连接失败" in str(exc_info.value)
    assert "provider=ollama" in str(exc_info.value)
    assert "model=llama3:8b" in str(exc_info.value)
    # 确保不泄露 base_url
    assert "localhost:11434" not in str(exc_info.value)
    assert "无法连接到 Ollama 服务" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_ollama_llm_timeout_error(mock_urlopen):
    """测试 Ollama LLM 处理超时错误"""
    # Mock 超时错误
    timeout_error = urllib.error.URLError("timed out")
    mock_urlopen.side_effect = timeout_error
    
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "Hello"}])
    
    assert "Ollama API 连接失败" in str(exc_info.value)
    assert "无法连接到 Ollama 服务" in str(exc_info.value)
    # 确保不泄露 base_url
    assert "localhost:11434" not in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_ollama_llm_invalid_response_format(mock_urlopen):
    """测试 Ollama LLM 处理无效响应格式"""
    # Mock 无效响应（缺少 message 字段）
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "model": "llama3:8b",
        "done": True
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    with pytest.raises(RuntimeError, match="API 响应格式错误"):
        llm.chat([{"role": "user", "content": "Hello"}])


@patch("urllib.request.urlopen")
def test_ollama_llm_missing_content_in_response(mock_urlopen):
    """测试 Ollama LLM 处理响应中缺少 content 字段"""
    # Mock 响应缺少 content 字段
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "model": "llama3:8b",
        "message": {
            "role": "assistant"
        },
        "done": True
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    with pytest.raises(RuntimeError, match="API 响应格式错误.*message.content"):
        llm.chat([{"role": "user", "content": "Hello"}])


@patch("urllib.request.urlopen")
def test_ollama_llm_with_system_message(mock_urlopen):
    """测试 Ollama LLM 支持 system 消息"""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "model": "llama3:8b",
        "message": {
            "role": "assistant",
            "content": "I'm a helpful assistant."
        },
        "done": True
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]
    response = llm.chat(messages)
    
    assert response == "I'm a helpful assistant."


@patch("urllib.request.urlopen")
def test_ollama_llm_multi_turn_conversation(mock_urlopen):
    """测试 Ollama LLM 支持多轮对话"""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "model": "llama3:8b",
        "message": {
            "role": "assistant",
            "content": "Python is a programming language."
        },
        "done": True
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://localhost:11434"
    ))
    
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more."}
    ]
    response = llm.chat(messages)
    
    assert response == "Python is a programming language."


def test_ollama_llm_default_base_url():
    """测试 Ollama LLM 使用默认 base_url"""
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b"
        # ollama_base_url 使用默认值 "http://localhost:11434"
    ))
    
    assert llm.get_provider() == "ollama"
    assert llm.get_model_name() == "llama3:8b"


def test_ollama_llm_custom_base_url():
    """测试 Ollama LLM 支持自定义 base_url"""
    llm = OllamaLLM(LLMConfig(
        provider="ollama",
        model="llama3:8b",
        ollama_base_url="http://192.168.1.100:11434"
    ))
    
    assert llm.get_provider() == "ollama"
    assert llm.get_model_name() == "llama3:8b"


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
