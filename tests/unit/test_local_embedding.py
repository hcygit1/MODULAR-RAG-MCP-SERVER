"""
Local Embedding 测试

测试 Local Embedding 实现（占位/适配层）。
"""
import pytest

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.local_embedding import LocalEmbedding


def test_factory_creates_local_embedding():
    """测试工厂可以创建 Local Embedding"""
    embedding_config = EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    )
    settings = _create_test_settings(embedding_config)
    
    embedding = EmbeddingFactory.create(settings)
    
    assert isinstance(embedding, LocalEmbedding)
    assert embedding.get_provider() == "local"
    assert embedding.get_model_name() == "bge-base-zh-v1.5"
    assert embedding.get_dimension() == 768


def test_local_embedding_missing_model():
    """测试 Local Embedding 缺少 model 时抛出错误"""
    with pytest.raises(ValueError, match="Local Embedding model 名称不能为空"):
        LocalEmbedding(EmbeddingConfig(
            provider="local",
            model="",
            local_model_path="",
            device="cpu"
        ))


def test_local_embedding_empty_texts():
    """测试 Local Embedding 处理空文本列表"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    with pytest.raises(ValueError, match="文本列表不能为空"):
        embedding.embed([])


def test_local_embedding_empty_string():
    """测试 Local Embedding 处理空字符串"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    with pytest.raises(ValueError, match="文本 0 不能为空"):
        embedding.embed([""])
    
    with pytest.raises(ValueError, match="文本 0 不能为空"):
        embedding.embed(["   "])  # 只有空白字符


def test_local_embedding_invalid_text_type():
    """测试 Local Embedding 处理无效文本类型"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    with pytest.raises(ValueError, match="文本 0 必须是字符串类型"):
        embedding.embed([123])  # 非字符串
    
    with pytest.raises(ValueError, match="文本 1 必须是字符串类型"):
        embedding.embed(["Hello", None])


def test_local_embedding_basic_functionality():
    """测试 Local Embedding 基本功能"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    texts = ["Hello", "World"]
    vectors = embedding.embed(texts)
    
    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert len(vectors[1]) == 768
    # 验证向量是浮点数列表
    assert all(isinstance(v, float) for v in vectors[0])
    assert all(isinstance(v, float) for v in vectors[1])


def test_local_embedding_stable_vectors():
    """测试 Local Embedding 生成稳定的向量（相同文本返回相同向量）"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    text = "Test text"
    vector1 = embedding.embed([text])[0]
    vector2 = embedding.embed([text])[0]
    
    # 相同文本应该返回相同的向量
    assert vector1 == vector2


def test_local_embedding_different_texts_different_vectors():
    """测试 Local Embedding 不同文本返回不同向量"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    texts = ["Text 1", "Text 2", "Text 3"]
    vectors = embedding.embed(texts)
    
    # 不同文本应该返回不同的向量
    assert len(set(tuple(v) for v in vectors)) == 3


def test_local_embedding_batch_processing():
    """测试 Local Embedding 批量处理"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    vectors = embedding.embed(texts)
    
    assert len(vectors) == 5
    assert all(len(v) == 768 for v in vectors)


def test_local_embedding_model_dimensions():
    """测试 Local Embedding 模型维度映射"""
    # bge-large-zh-v1.5
    embedding1 = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-large-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    assert embedding1.get_dimension() == 1024
    
    # bge-base-zh-v1.5
    embedding2 = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    assert embedding2.get_dimension() == 768
    
    # bge-small-zh-v1.5
    embedding3 = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-small-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    assert embedding3.get_dimension() == 512
    
    # 未知模型（使用默认维度 768）
    embedding4 = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="unknown-model",
        local_model_path="",
        device="cpu"
    ))
    assert embedding4.get_dimension() == 768  # 默认维度


def test_local_embedding_case_insensitive_model_name():
    """测试 Local Embedding 模型名称大小写不敏感"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="BGE-BASE-ZH-V1.5",  # 大写
        local_model_path="",
        device="cpu"
    ))
    assert embedding.get_dimension() == 768


def test_local_embedding_trace_parameter():
    """测试 Local Embedding 接受 trace 参数（即使暂未使用）"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    # trace 参数应该被接受（即使为 None）
    texts = ["Hello"]
    vectors = embedding.embed(texts, trace=None)
    assert len(vectors) == 1


def test_local_embedding_vector_range():
    """测试 Local Embedding 向量值在合理范围内"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="",
        device="cpu"
    ))
    
    texts = ["Test"]
    vectors = embedding.embed(texts)
    
    # 向量值应该在 [-1, 1] 范围内（基于哈希归一化）
    assert all(-1.0 <= v <= 1.0 for v in vectors[0])


def test_local_embedding_config_storage():
    """测试 Local Embedding 存储配置信息"""
    embedding = LocalEmbedding(EmbeddingConfig(
        provider="local",
        model="bge-base-zh-v1.5",
        local_model_path="/path/to/model",
        device="cuda"
    ))
    
    # 验证配置信息被存储（虽然当前占位实现不使用）
    assert embedding._model_path == "/path/to/model"
    assert embedding._device == "cuda"


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
