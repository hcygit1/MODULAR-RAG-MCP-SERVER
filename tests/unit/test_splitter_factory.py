"""
Splitter Factory 测试

测试 SplitterFactory 的路由逻辑和错误处理。
"""
import pytest

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.splitter.fake_splitter import FakeSplitter


def test_factory_creates_fake_splitter():
    """测试工厂可以创建 Fake Splitter"""
    fake_splitter = SplitterFactory.create_fake(
        strategy="test", chunk_size=256, chunk_overlap=32
    )
    
    assert isinstance(fake_splitter, BaseSplitter)
    assert fake_splitter.get_strategy() == "test"
    assert fake_splitter.get_chunk_size() == 256
    assert fake_splitter.get_chunk_overlap() == 32


def test_fake_splitter_split_text():
    """测试 Fake Splitter 的 split_text 方法"""
    fake_splitter = FakeSplitter(chunk_size=10, chunk_overlap=2)
    
    text = "This is a test text for splitting."
    chunks = fake_splitter.split_text(text)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    
    # 验证所有块的总长度（考虑重叠）
    total_length = sum(len(chunk) for chunk in chunks)
    assert total_length >= len(text)


def test_fake_splitter_empty_text():
    """测试 Fake Splitter 处理空文本"""
    fake_splitter = FakeSplitter()
    
    with pytest.raises(ValueError, match="输入文本不能为空"):
        fake_splitter.split_text("")


def test_fake_splitter_short_text():
    """测试 Fake Splitter 处理短文本（小于 chunk_size）"""
    fake_splitter = FakeSplitter(chunk_size=100, chunk_overlap=10)
    
    text = "Short text"
    chunks = fake_splitter.split_text(text)
    
    # 短文本应该只返回一个块
    assert len(chunks) == 1
    assert chunks[0] == text


def test_fake_splitter_chunk_size():
    """测试 Fake Splitter 的块大小"""
    fake_splitter = FakeSplitter(chunk_size=20, chunk_overlap=5)
    
    # 创建一个足够长的文本
    text = "A" * 100
    chunks = fake_splitter.split_text(text)
    
    # 验证每个块的大小（最后一个块可能小于 chunk_size）
    for chunk in chunks[:-1]:
        assert len(chunk) == 20
    
    # 最后一个块应该小于等于 chunk_size
    assert len(chunks[-1]) <= 20


def test_fake_splitter_chunk_overlap():
    """测试 Fake Splitter 的块重叠"""
    fake_splitter = FakeSplitter(chunk_size=10, chunk_overlap=3)
    
    text = "A" * 30
    chunks = fake_splitter.split_text(text)
    
    # 验证相邻块之间有重叠
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # 检查是否有重叠部分
            # 下一个块的开始部分应该与当前块的结束部分相同
            overlap = current_chunk[-3:]
            assert next_chunk.startswith(overlap)


def test_factory_unsupported_strategy():
    """测试不支持的策略抛出 ValueError"""
    settings = _create_test_settings()
    
    with pytest.raises(ValueError, match="不支持的 Splitter 策略"):
        SplitterFactory.create(settings, strategy="unsupported")


def test_factory_not_implemented_strategies():
    """测试尚未实现的策略抛出 NotImplementedError"""
    settings = _create_test_settings()
    
    # 测试 recursive
    with pytest.raises(NotImplementedError, match="Recursive Splitter 实现将在 B7.5"):
        SplitterFactory.create(settings, strategy="recursive")
    
    # 测试 semantic
    with pytest.raises(NotImplementedError, match="Semantic Splitter 实现尚未完成"):
        SplitterFactory.create(settings, strategy="semantic")
    
    # 测试 fixed
    with pytest.raises(NotImplementedError, match="Fixed Splitter 实现尚未完成"):
        SplitterFactory.create(settings, strategy="fixed")


def test_factory_strategy_case_insensitive():
    """测试策略名称大小写不敏感"""
    settings = _create_test_settings()
    
    # 测试大写
    with pytest.raises(NotImplementedError, match="Recursive Splitter"):
        SplitterFactory.create(settings, strategy="RECURSIVE")
    
    # 测试混合大小写
    with pytest.raises(NotImplementedError, match="Recursive Splitter"):
        SplitterFactory.create(settings, strategy="ReCuRsIvE")


def test_base_splitter_interface():
    """测试 BaseSplitter 接口定义"""
    # 验证 BaseSplitter 是抽象类
    with pytest.raises(TypeError):
        BaseSplitter()  # 不能直接实例化抽象类
    
    # 验证 FakeSplitter 实现了所有抽象方法
    fake_splitter = FakeSplitter()
    assert hasattr(fake_splitter, "split_text")
    assert hasattr(fake_splitter, "get_strategy")
    assert hasattr(fake_splitter, "get_chunk_size")
    assert hasattr(fake_splitter, "get_chunk_overlap")
    
    # 验证方法可以调用
    chunks = fake_splitter.split_text("test")
    assert isinstance(chunks, list)
    assert fake_splitter.get_strategy() == "fake"
    assert fake_splitter.get_chunk_size() == 512
    assert fake_splitter.get_chunk_overlap() == 50


def test_fake_splitter_trace_parameter():
    """测试 Fake Splitter 接受 trace 参数（虽然不使用）"""
    fake_splitter = FakeSplitter()
    
    # trace 参数是可选的，应该可以传入 None
    chunks1 = fake_splitter.split_text("test", trace=None)
    
    # 或者不传入（使用默认值）
    chunks2 = fake_splitter.split_text("test")
    
    # 结果应该相同
    assert chunks1 == chunks2


def test_fake_splitter_different_configs():
    """测试 Fake Splitter 的不同配置"""
    # 测试不同的 chunk_size
    splitter1 = FakeSplitter(chunk_size=100, chunk_overlap=10)
    splitter2 = FakeSplitter(chunk_size=200, chunk_overlap=20)
    
    text = "A" * 500
    chunks1 = splitter1.split_text(text)
    chunks2 = splitter2.split_text(text)
    
    # chunk_size 更大的应该产生更少的块
    assert len(chunks2) <= len(chunks1)
    
    # 验证配置正确
    assert splitter1.get_chunk_size() == 100
    assert splitter2.get_chunk_size() == 200


def _create_test_settings() -> Settings:
    """创建测试用的 Settings 对象"""
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
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
