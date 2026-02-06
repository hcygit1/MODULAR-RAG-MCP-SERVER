"""
LLM Factory 测试

测试 LLMFactory 的路由逻辑和错误处理。
"""
import pytest
from dataclasses import dataclass

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.fake_llm import FakeLLM


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


def test_factory_not_implemented_providers():
    """测试尚未实现的 provider 抛出 NotImplementedError"""
    # 测试 azure
    llm_config_azure = LLMConfig(provider="azure", model="gpt-4o")
    settings_azure = Settings(
        llm=llm_config_azure,
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(
            backend="chroma", persist_path="./data/db/chroma", collection_name="test"
        ),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10
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
            chunk_size=512, chunk_overlap=50,
            enable_llm_refinement=False, enable_metadata_enrichment=True,
            enable_image_captioning=True, batch_size=32
        )
    )
    
    with pytest.raises(NotImplementedError, match="Azure LLM 实现将在 B7.1"):
        LLMFactory.create(settings_azure)
    
    # 测试 openai
    llm_config_openai = LLMConfig(provider="openai", model="gpt-4o")
    settings_openai = Settings(
        llm=llm_config_openai,
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(
            backend="chroma", persist_path="./data/db/chroma", collection_name="test"
        ),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10
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
            chunk_size=512, chunk_overlap=50,
            enable_llm_refinement=False, enable_metadata_enrichment=True,
            enable_image_captioning=True, batch_size=32
        )
    )
    
    with pytest.raises(NotImplementedError, match="OpenAI LLM 实现将在 B7.1"):
        LLMFactory.create(settings_openai)
    
    # 测试 ollama
    llm_config_ollama = LLMConfig(provider="ollama", model="llama-3-8b")
    settings_ollama = Settings(
        llm=llm_config_ollama,
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(
            backend="chroma", persist_path="./data/db/chroma", collection_name="test"
        ),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10
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
            chunk_size=512, chunk_overlap=50,
            enable_llm_refinement=False, enable_metadata_enrichment=True,
            enable_image_captioning=True, batch_size=32
        )
    )
    
    with pytest.raises(NotImplementedError, match="Ollama LLM 实现将在 B7.2"):
        LLMFactory.create(settings_ollama)
    
    # 测试 deepseek
    llm_config_deepseek = LLMConfig(provider="deepseek", model="deepseek-chat")
    settings_deepseek = Settings(
        llm=llm_config_deepseek,
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(
            backend="chroma", persist_path="./data/db/chroma", collection_name="test"
        ),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10
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
            chunk_size=512, chunk_overlap=50,
            enable_llm_refinement=False, enable_metadata_enrichment=True,
            enable_image_captioning=True, batch_size=32
        )
    )
    
    with pytest.raises(NotImplementedError, match="DeepSeek LLM 实现将在 B7.1"):
        LLMFactory.create(settings_deepseek)


def test_factory_provider_case_insensitive():
    """测试 provider 名称大小写不敏感"""
    # 测试大写
    llm_config_upper = LLMConfig(provider="AZURE", model="gpt-4o")
    settings_upper = Settings(
        llm=llm_config_upper,
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(
            backend="chroma", persist_path="./data/db/chroma", collection_name="test"
        ),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10
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
            chunk_size=512, chunk_overlap=50,
            enable_llm_refinement=False, enable_metadata_enrichment=True,
            enable_image_captioning=True, batch_size=32
        )
    )
    
    # 应该能识别（虽然会抛出 NotImplementedError，但错误信息应该正确）
    with pytest.raises(NotImplementedError, match="Azure LLM"):
        LLMFactory.create(settings_upper)


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
