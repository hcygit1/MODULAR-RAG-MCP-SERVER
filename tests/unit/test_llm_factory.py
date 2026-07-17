"""
LLM Factory 测试

测试 LLMFactory 的路由逻辑和错误处理。
"""

import pytest

from modular_rag_mcp_server.core.settings import (
    DashboardConfig,
    EmbeddingConfig,
    EvaluationConfig,
    IngestionConfig,
    LLMConfig,
    LoggingConfig,
    ObservabilityConfig,
    RerankConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
    VisionLLMConfig,
)
from modular_rag_mcp_server.libs.llm.azure_llm import AzureLLM
from modular_rag_mcp_server.libs.llm.base_llm import BaseLLM
from modular_rag_mcp_server.libs.llm.deepseek_llm import DeepSeekLLM
from modular_rag_mcp_server.libs.llm.fake_llm import FakeLLM
from modular_rag_mcp_server.libs.llm.llm_factory import LLMFactory
from modular_rag_mcp_server.libs.llm.ollama_llm import OllamaLLM
from modular_rag_mcp_server.libs.llm.openai_llm import OpenAILLM


def test_factory_creates_fake_llm():
    """测试工厂可以创建 Fake LLM"""
    fake_llm = LLMFactory.create_fake(provider="test", model="test-model")
    
    assert isinstance(fake_llm, BaseLLM)
    assert fake_llm.get_provider() == "test"
    assert fake_llm.get_model_name() == "test-model"


def test_fake_llm_chat():
    """测试 Fake LLM 的 chat 方法"""
    fake_llm = FakeLLM(provider="fake", model="fake-model")
    
    messages = [{"role": "user", "content": "Hello"}]
    response = fake_llm.chat(messages)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert "Hello" in response


def test_fake_llm_empty_messages():
    """测试 Fake LLM 处理空消息列表"""
    fake_llm = FakeLLM()
    
    with pytest.raises(ValueError, match="消息列表不能为空"):
        fake_llm.chat([])


def test_factory_unsupported_provider():
    """测试不支持的 provider 抛出 ValueError"""
    # 创建一个包含不支持 provider 的配置
    llm_config = LLMConfig(
        provider="unsupported",
        model="test-model"
    )
    
    # 创建完整的 Settings 对象（使用最小配置）
    settings = Settings(
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
        rerank=RerankConfig(
            backend="none",
            model="",
            top_m=30,
            timeout_seconds=5
        ),
        evaluation=EvaluationConfig(
            backends=["custom"],
            golden_test_set="./tests/fixtures/golden_test_set.json"
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
    
    with pytest.raises(ValueError, match="不支持的 LLM provider"):
        LLMFactory.create(settings)


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (
            LLMConfig(
                provider="azure",
                model="gpt-4o",
                azure_endpoint="https://example.openai.azure.com",
                azure_api_key="test-key",
                deployment_name="test-deployment",
            ),
            AzureLLM,
        ),
        (
            LLMConfig(
                provider="openai",
                model="gpt-4o",
                openai_api_key="test-key",
            ),
            OpenAILLM,
        ),
        (
            LLMConfig(provider="ollama", model="llama-3-8b"),
            OllamaLLM,
        ),
        (
            LLMConfig(
                provider="deepseek",
                model="deepseek-chat",
                deepseek_api_key="test-key",
            ),
            DeepSeekLLM,
        ),
    ],
)
def test_factory_creates_implemented_providers(config, expected_type):
    """测试已实现 provider 能被工厂正确路由。"""
    assert isinstance(LLMFactory.create(config=config), expected_type)


def test_factory_provider_case_insensitive():
    """测试 provider 名称大小写不敏感"""
    config = LLMConfig(provider="OLLAMA", model="llama-3-8b")
    
    llm = LLMFactory.create(config=config)

    assert isinstance(llm, OllamaLLM)
    assert llm.get_provider() == "ollama"


def test_base_llm_interface():
    """测试 BaseLLM 接口定义"""
    # 验证 BaseLLM 是抽象类
    with pytest.raises(TypeError):
        BaseLLM()  # 不能直接实例化抽象类
    
    # 验证 FakeLLM 实现了所有抽象方法
    fake_llm = FakeLLM()
    assert hasattr(fake_llm, "chat")
    assert hasattr(fake_llm, "get_model_name")
    assert hasattr(fake_llm, "get_provider")
    
    # 验证方法可以调用
    response = fake_llm.chat([{"role": "user", "content": "test"}])
    assert isinstance(response, str)
    assert fake_llm.get_model_name() == "fake-model"
    assert fake_llm.get_provider() == "fake"
