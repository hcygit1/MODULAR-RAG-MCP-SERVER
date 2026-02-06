"""
Embedding Factory 测试

测试 EmbeddingFactory 的路由逻辑和错误处理。
"""
import pytest

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.fake_embedding import FakeEmbedding


def test_factory_creates_fake_embedding():
    """测试工厂可以创建 Fake Embedding"""
    fake_embedding = EmbeddingFactory.create_fake(
        provider="test", model="test-model", dimension=256
    )
    
    assert isinstance(fake_embedding, BaseEmbedding)
    assert fake_embedding.get_provider() == "test"
    assert fake_embedding.get_model_name() == "test-model"
    assert fake_embedding.get_dimension() == 256


def test_fake_embedding_embed():
    """测试 Fake Embedding 的 embed 方法"""
    fake_embedding = FakeEmbedding(provider="fake", model="fake-model", dimension=128)
    
    texts = ["Hello world", "Test embedding"]
    vectors = fake_embedding.embed(texts)
    
    assert len(vectors) == 2
    assert len(vectors[0]) == 128
    assert len(vectors[1]) == 128
    assert all(isinstance(v, float) for v in vectors[0])
    assert all(isinstance(v, float) for v in vectors[1])


def test_fake_embedding_stable_vectors():
    """测试 Fake Embedding 返回稳定向量（相同文本返回相同向量）"""
    fake_embedding = FakeEmbedding(dimension=64)
    
    text = "Stable vector test"
    vectors1 = fake_embedding.embed([text])
    vectors2 = fake_embedding.embed([text])
    
    # 相同文本应该返回相同的向量
    assert vectors1[0] == vectors2[0]
    
    # 不同文本应该返回不同的向量
    vectors3 = fake_embedding.embed(["Different text"])
    assert vectors1[0] != vectors3[0]


def test_fake_embedding_empty_texts():
    """测试 Fake Embedding 处理空文本列表"""
    fake_embedding = FakeEmbedding()
    
    with pytest.raises(ValueError, match="文本列表不能为空"):
        fake_embedding.embed([])


def test_fake_embedding_batch():
    """测试 Fake Embedding 批量处理"""
    fake_embedding = FakeEmbedding(dimension=32)
    
    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    vectors = fake_embedding.embed(texts)
    
    assert len(vectors) == 5
    assert all(len(v) == 32 for v in vectors)
    # 确保每个向量都不同
    assert len(set(tuple(v) for v in vectors)) == 5


def test_factory_unsupported_provider():
    """测试不支持的 provider 抛出 ValueError"""
    embedding_config = EmbeddingConfig(
        provider="unsupported",
        model="test-model"
    )
    
    settings = _create_test_settings(embedding_config)
    
    with pytest.raises(ValueError, match="不支持的 Embedding provider"):
        EmbeddingFactory.create(settings)


def test_factory_not_implemented_providers():
    """测试尚未实现的 provider 抛出 NotImplementedError"""
    # 测试 openai
    embedding_config_openai = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small"
    )
    settings_openai = _create_test_settings(embedding_config_openai)
    
    with pytest.raises(NotImplementedError, match="OpenAI Embedding 实现将在 B7.3"):
        EmbeddingFactory.create(settings_openai)
    
    # 测试 local
    embedding_config_local = EmbeddingConfig(
        provider="local",
        model="bge-large-zh-v1.5",
        local_model_path="./models/bge"
    )
    settings_local = _create_test_settings(embedding_config_local)
    
    with pytest.raises(NotImplementedError, match="Local Embedding 实现将在 B7.4"):
        EmbeddingFactory.create(settings_local)
    
    # 测试 ollama
    embedding_config_ollama = EmbeddingConfig(
        provider="ollama",
        model="nomic-embed-text"
    )
    settings_ollama = _create_test_settings(embedding_config_ollama)
    
    with pytest.raises(NotImplementedError, match="Ollama Embedding 实现将在 B7.4"):
        EmbeddingFactory.create(settings_ollama)


def test_factory_provider_case_insensitive():
    """测试 provider 名称大小写不敏感"""
    embedding_config_upper = EmbeddingConfig(
        provider="OPENAI",
        model="text-embedding-3-small"
    )
    settings_upper = _create_test_settings(embedding_config_upper)
    
    # 应该能识别（虽然会抛出 NotImplementedError，但错误信息应该正确）
    with pytest.raises(NotImplementedError, match="OpenAI Embedding"):
        EmbeddingFactory.create(settings_upper)


def test_base_embedding_interface():
    """测试 BaseEmbedding 接口定义"""
    # 验证 BaseEmbedding 是抽象类
    with pytest.raises(TypeError):
        BaseEmbedding()  # 不能直接实例化抽象类
    
    # 验证 FakeEmbedding 实现了所有抽象方法
    fake_embedding = FakeEmbedding()
    assert hasattr(fake_embedding, "embed")
    assert hasattr(fake_embedding, "get_model_name")
    assert hasattr(fake_embedding, "get_provider")
    assert hasattr(fake_embedding, "get_dimension")
    
    # 验证方法可以调用
    vectors = fake_embedding.embed(["test"])
    assert isinstance(vectors, list)
    assert fake_embedding.get_model_name() == "fake-model"
    assert fake_embedding.get_provider() == "fake"
    assert fake_embedding.get_dimension() == 128


def test_fake_embedding_dimension():
    """测试 Fake Embedding 的不同维度"""
    for dim in [64, 128, 256, 512, 1536]:
        fake_embedding = FakeEmbedding(dimension=dim)
        assert fake_embedding.get_dimension() == dim
        
        vectors = fake_embedding.embed(["test"])
        assert len(vectors[0]) == dim


def test_fake_embedding_trace_parameter():
    """测试 Fake Embedding 接受 trace 参数（虽然不使用）"""
    fake_embedding = FakeEmbedding()
    
    # trace 参数是可选的，应该可以传入 None
    vectors1 = fake_embedding.embed(["test"], trace=None)
    
    # 或者不传入（使用默认值）
    vectors2 = fake_embedding.embed(["test"])
    
    # 结果应该相同
    assert vectors1 == vectors2


def _create_test_settings(embedding_config: EmbeddingConfig) -> Settings:
    """创建测试用的 Settings 对象"""
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
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
