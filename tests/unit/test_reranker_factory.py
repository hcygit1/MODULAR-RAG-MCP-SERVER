"""
Reranker Factory 测试

测试 RerankerFactory 的路由逻辑和错误处理。
"""
import pytest

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.reranker_factory import RerankerFactory
from src.libs.reranker.none_reranker import NoneReranker
from src.libs.vector_store.base_vector_store import QueryResult


def test_factory_creates_none_reranker():
    """测试工厂可以创建 NoneReranker"""
    none_reranker = RerankerFactory.create_none()
    
    assert isinstance(none_reranker, BaseReranker)
    assert none_reranker.get_backend() == "none"


def test_none_reranker_no_change():
    """测试 NoneReranker 不改变排序"""
    reranker = NoneReranker()
    
    query = "test query"
    candidates = [
        QueryResult(id="id1", score=0.9, text="Text 1", metadata={}),
        QueryResult(id="id2", score=0.8, text="Text 2", metadata={}),
        QueryResult(id="id3", score=0.7, text="Text 3", metadata={})
    ]
    
    result = reranker.rerank(query, candidates)
    
    # 验证顺序没有改变
    assert len(result) == 3
    assert result[0].id == "id1"
    assert result[1].id == "id2"
    assert result[2].id == "id3"
    
    # 验证分数没有改变
    assert result[0].score == 0.9
    assert result[1].score == 0.8
    assert result[2].score == 0.7


def test_none_reranker_empty_candidates():
    """测试 NoneReranker 处理空候选列表"""
    reranker = NoneReranker()
    
    with pytest.raises(ValueError, match="候选列表不能为空"):
        reranker.rerank("test query", [])


def test_none_reranker_returns_copy():
    """测试 NoneReranker 返回列表的副本（不修改原列表）"""
    reranker = NoneReranker()
    
    candidates = [
        QueryResult(id="id1", score=0.9, text="Text 1", metadata={})
    ]
    
    result = reranker.rerank("test query", candidates)
    
    # 修改结果不应该影响原列表
    result[0].score = 0.5
    assert candidates[0].score == 0.9  # 原列表未改变


def test_factory_create_none_backend():
    """测试工厂创建 none backend 的 Reranker"""
    rerank_config = RerankConfig(
        backend="none",
        model="",
        top_m=30,
        timeout_seconds=5
    )
    
    settings = _create_test_settings(rerank_config)
    reranker = RerankerFactory.create(settings)
    
    assert isinstance(reranker, NoneReranker)
    assert reranker.get_backend() == "none"


def test_factory_create_empty_backend():
    """测试工厂创建空 backend（默认为 none）"""
    rerank_config = RerankConfig(
        backend="",
        model="",
        top_m=30,
        timeout_seconds=5
    )
    
    settings = _create_test_settings(rerank_config)
    reranker = RerankerFactory.create(settings)
    
    assert isinstance(reranker, NoneReranker)
    assert reranker.get_backend() == "none"


def test_factory_unsupported_backend():
    """测试不支持的 backend 抛出 ValueError"""
    rerank_config = RerankConfig(
        backend="unsupported",
        model="",
        top_m=30,
        timeout_seconds=5
    )
    
    settings = _create_test_settings(rerank_config)
    
    with pytest.raises(ValueError, match="不支持的 Reranker backend"):
        RerankerFactory.create(settings)


def test_factory_not_implemented_backends():
    """测试尚未实现的 backend 抛出 NotImplementedError"""
    # 测试 cross_encoder
    rerank_config_cross_encoder = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )
    settings_cross_encoder = _create_test_settings(rerank_config_cross_encoder)
    
    with pytest.raises(NotImplementedError, match="CrossEncoder Reranker 实现将在 B7.8"):
        RerankerFactory.create(settings_cross_encoder)
    
    # 测试 llm
    rerank_config_llm = RerankConfig(
        backend="llm",
        model="gpt-4o",
        top_m=30,
        timeout_seconds=5
    )
    settings_llm = _create_test_settings(rerank_config_llm)
    
    with pytest.raises(NotImplementedError, match="LLM Reranker 实现将在 B7.7"):
        RerankerFactory.create(settings_llm)


def test_factory_backend_case_insensitive():
    """测试 backend 名称大小写不敏感"""
    # 测试大写
    rerank_config_upper = RerankConfig(
        backend="NONE",
        model="",
        top_m=30,
        timeout_seconds=5
    )
    settings_upper = _create_test_settings(rerank_config_upper)
    
    reranker = RerankerFactory.create(settings_upper)
    assert isinstance(reranker, NoneReranker)
    
    # 测试混合大小写
    rerank_config_mixed = RerankConfig(
        backend="Cross_Encoder",
        model="test",
        top_m=30,
        timeout_seconds=5
    )
    settings_mixed = _create_test_settings(rerank_config_mixed)
    
    with pytest.raises(NotImplementedError, match="CrossEncoder Reranker"):
        RerankerFactory.create(settings_mixed)


def test_base_reranker_interface():
    """测试 BaseReranker 接口定义"""
    # 验证 BaseReranker 是抽象类
    with pytest.raises(TypeError):
        BaseReranker()  # 不能直接实例化抽象类
    
    # 验证 NoneReranker 实现了所有抽象方法
    none_reranker = NoneReranker()
    assert hasattr(none_reranker, "rerank")
    assert hasattr(none_reranker, "get_backend")
    
    # 验证方法可以调用
    candidates = [QueryResult(id="test", score=0.9, text="test", metadata={})]
    result = none_reranker.rerank("test query", candidates)
    assert isinstance(result, list)
    assert none_reranker.get_backend() == "none"


def test_none_reranker_trace_parameter():
    """测试 NoneReranker 接受 trace 参数（虽然不使用）"""
    reranker = NoneReranker()
    
    candidates = [QueryResult(id="test", score=0.9, text="test", metadata={})]
    
    # trace 参数是可选的，应该可以传入 None
    result1 = reranker.rerank("test query", candidates, trace=None)
    
    # 或者不传入（使用默认值）
    result2 = reranker.rerank("test query", candidates)
    
    # 结果应该相同
    assert len(result1) == len(result2)
    assert result1[0].id == result2[0].id


def test_none_reranker_preserves_metadata():
    """测试 NoneReranker 保留元数据"""
    reranker = NoneReranker()
    
    candidates = [
        QueryResult(
            id="id1",
            score=0.9,
            text="Text 1",
            metadata={"source": "doc1.pdf", "page": 1}
        )
    ]
    
    result = reranker.rerank("test query", candidates)
    
    assert result[0].metadata == {"source": "doc1.pdf", "page": 1}


def test_none_reranker_single_candidate():
    """测试 NoneReranker 处理单个候选"""
    reranker = NoneReranker()
    
    candidates = [QueryResult(id="id1", score=0.9, text="Text 1", metadata={})]
    result = reranker.rerank("test query", candidates)
    
    assert len(result) == 1
    assert result[0].id == "id1"


def _create_test_settings(rerank_config: RerankConfig) -> Settings:
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
        rerank=rerank_config,
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
