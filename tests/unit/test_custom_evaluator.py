"""
Custom Evaluator 测试

测试 CustomEvaluator 的评估指标计算和错误处理。
"""
import pytest

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.libs.evaluator.custom_evaluator import CustomEvaluator


def test_factory_creates_custom_evaluator():
    """测试工厂可以创建 CustomEvaluator"""
    custom_evaluator = EvaluatorFactory.create_custom()
    
    assert isinstance(custom_evaluator, BaseEvaluator)
    assert custom_evaluator.get_backend() == "custom"


def test_custom_evaluator_hit_rate_perfect():
    """测试 CustomEvaluator 计算完美命中率"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2", "id3"]
    golden_ids = ["id1", "id2"]
    
    metrics = evaluator.evaluate(query, retrieved_ids, golden_ids)
    
    assert "hit_rate" in metrics
    assert metrics["hit_rate"] == 1.0  # 所有标准答案都在检索结果中


def test_custom_evaluator_hit_rate_partial():
    """测试 CustomEvaluator 计算部分命中率"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2", "id3"]
    golden_ids = ["id1", "id4"]  # id4 不在检索结果中
    
    metrics = evaluator.evaluate(query, retrieved_ids, golden_ids)
    
    assert metrics["hit_rate"] == 0.5  # 2个标准答案中只有1个命中


def test_custom_evaluator_hit_rate_zero():
    """测试 CustomEvaluator 计算零命中率"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2", "id3"]
    golden_ids = ["id4", "id5"]  # 都不在检索结果中
    
    metrics = evaluator.evaluate(query, retrieved_ids, golden_ids)
    
    assert metrics["hit_rate"] == 0.0


def test_custom_evaluator_mrr_perfect():
    """测试 CustomEvaluator 计算完美 MRR（第一个结果就是标准答案）"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2", "id3"]
    golden_ids = ["id1"]
    
    metrics = evaluator.evaluate(query, retrieved_ids, golden_ids)
    
    assert "mrr" in metrics
    assert metrics["mrr"] == 1.0  # 第一个结果就是标准答案，MRR = 1/1 = 1.0


def test_custom_evaluator_mrr_second():
    """测试 CustomEvaluator 计算 MRR（第二个结果才是标准答案）"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2", "id3"]
    golden_ids = ["id2"]
    
    metrics = evaluator.evaluate(query, retrieved_ids, golden_ids)
    
    assert metrics["mrr"] == 0.5  # 第二个结果才是标准答案，MRR = 1/2 = 0.5


def test_custom_evaluator_mrr_zero():
    """测试 CustomEvaluator 计算零 MRR（没有标准答案在检索结果中）"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2", "id3"]
    golden_ids = ["id4"]
    
    metrics = evaluator.evaluate(query, retrieved_ids, golden_ids)
    
    assert metrics["mrr"] == 0.0


def test_custom_evaluator_mrr_multiple_golden():
    """测试 CustomEvaluator MRR（多个标准答案，取第一个出现的位置）"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2", "id3"]
    golden_ids = ["id3", "id1"]  # id3 在第三个位置，id1 在第一个位置
    
    metrics = evaluator.evaluate(query, retrieved_ids, golden_ids)
    
    # MRR 应该取第一个出现的标准答案的位置
    # id1 在第一个位置，所以 MRR = 1/1 = 1.0
    assert metrics["mrr"] == 1.0


def test_custom_evaluator_empty_retrieved():
    """测试 CustomEvaluator 处理空检索结果"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = []
    golden_ids = ["id1"]
    
    with pytest.raises(ValueError, match="检索结果列表不能为空"):
        evaluator.evaluate(query, retrieved_ids, golden_ids)


def test_custom_evaluator_empty_golden():
    """测试 CustomEvaluator 处理空标准答案"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2"]
    golden_ids = []
    
    with pytest.raises(ValueError, match="标准答案列表不能为空"):
        evaluator.evaluate(query, retrieved_ids, golden_ids)


def test_custom_evaluator_stable_metrics():
    """测试 CustomEvaluator 返回稳定的 metrics（相同输入返回相同输出）"""
    evaluator = CustomEvaluator()
    
    query = "test query"
    retrieved_ids = ["id1", "id2", "id3"]
    golden_ids = ["id1", "id2"]
    
    metrics1 = evaluator.evaluate(query, retrieved_ids, golden_ids)
    metrics2 = evaluator.evaluate(query, retrieved_ids, golden_ids)
    
    assert metrics1 == metrics2
    assert metrics1["hit_rate"] == metrics2["hit_rate"]
    assert metrics1["mrr"] == metrics2["mrr"]


def test_base_evaluator_interface():
    """测试 BaseEvaluator 接口定义"""
    # 验证 BaseEvaluator 是抽象类
    with pytest.raises(TypeError):
        BaseEvaluator()  # 不能直接实例化抽象类
    
    # 验证 CustomEvaluator 实现了所有抽象方法
    custom_evaluator = CustomEvaluator()
    assert hasattr(custom_evaluator, "evaluate")
    assert hasattr(custom_evaluator, "get_backend")
    
    # 验证方法可以调用
    metrics = custom_evaluator.evaluate("test", ["id1"], ["id1"])
    assert isinstance(metrics, dict)
    assert custom_evaluator.get_backend() == "custom"


def test_factory_create_custom_backend():
    """测试工厂创建 custom backend 的 Evaluator"""
    evaluation_config = EvaluationConfig(
        backends=["custom"],
        golden_test_set="./tests/fixtures/golden_test_set.json"
    )
    
    settings = _create_test_settings(evaluation_config)
    evaluators = EvaluatorFactory.create(settings)
    
    assert len(evaluators) == 1
    assert isinstance(evaluators[0], CustomEvaluator)
    assert evaluators[0].get_backend() == "custom"


def test_factory_create_multiple_backends():
    """测试工厂创建多个 backend 的 Evaluator（组合模式）"""
    evaluation_config = EvaluationConfig(
        backends=["custom", "custom"],  # 可以重复
        golden_test_set="./tests/fixtures/golden_test_set.json"
    )
    
    settings = _create_test_settings(evaluation_config)
    evaluators = EvaluatorFactory.create(settings)
    
    assert len(evaluators) == 2
    assert all(isinstance(e, CustomEvaluator) for e in evaluators)


def test_factory_unsupported_backend():
    """测试不支持的 backend 抛出 ValueError"""
    evaluation_config = EvaluationConfig(
        backends=["unsupported"],
        golden_test_set="./tests/fixtures/golden_test_set.json"
    )
    
    settings = _create_test_settings(evaluation_config)
    
    with pytest.raises(ValueError, match="不支持的 Evaluator backend"):
        EvaluatorFactory.create(settings)


def test_factory_not_implemented_backends():
    """测试尚未实现的 backend 抛出 NotImplementedError"""
    # 测试 ragas
    evaluation_config_ragas = EvaluationConfig(
        backends=["ragas"],
        golden_test_set="./tests/fixtures/golden_test_set.json"
    )
    settings_ragas = _create_test_settings(evaluation_config_ragas)
    
    with pytest.raises(NotImplementedError, match="Ragas Evaluator 实现尚未完成"):
        EvaluatorFactory.create(settings_ragas)
    
    # 测试 deepeval
    evaluation_config_deepeval = EvaluationConfig(
        backends=["deepeval"],
        golden_test_set="./tests/fixtures/golden_test_set.json"
    )
    settings_deepeval = _create_test_settings(evaluation_config_deepeval)
    
    with pytest.raises(NotImplementedError, match="DeepEval Evaluator 实现尚未完成"):
        EvaluatorFactory.create(settings_deepeval)


def test_factory_backend_case_insensitive():
    """测试 backend 名称大小写不敏感"""
    evaluation_config_upper = EvaluationConfig(
        backends=["CUSTOM"],
        golden_test_set="./tests/fixtures/golden_test_set.json"
    )
    settings_upper = _create_test_settings(evaluation_config_upper)
    
    evaluators = EvaluatorFactory.create(settings_upper)
    assert len(evaluators) == 1
    assert isinstance(evaluators[0], CustomEvaluator)


def test_custom_evaluator_metrics_range():
    """测试 CustomEvaluator 返回的指标在合理范围内"""
    evaluator = CustomEvaluator()
    
    # 测试各种场景
    test_cases = [
        (["id1"], ["id1"], 1.0, 1.0),  # 完美匹配
        (["id1", "id2"], ["id1"], 1.0, 1.0),  # 第一个匹配
        (["id1", "id2"], ["id2"], 1.0, 0.5),  # 第二个匹配
        (["id1", "id2"], ["id3"], 0.0, 0.0),  # 无匹配
        (["id1", "id2", "id3"], ["id1", "id2"], 1.0, 1.0),  # 全部匹配
        (["id1", "id2", "id3"], ["id1", "id4"], 0.5, 1.0),  # 部分匹配
    ]
    
    for retrieved_ids, golden_ids, expected_hit_rate, expected_mrr in test_cases:
        metrics = evaluator.evaluate("test", retrieved_ids, golden_ids)
        assert 0.0 <= metrics["hit_rate"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0
        assert metrics["hit_rate"] == expected_hit_rate
        assert metrics["mrr"] == expected_mrr


def _create_test_settings(evaluation_config: EvaluationConfig) -> Settings:
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
        evaluation=evaluation_config,
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
