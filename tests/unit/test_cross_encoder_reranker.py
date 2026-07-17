"""
Cross-Encoder Reranker 单元测试

使用 mock scorer 测试 Cross-Encoder Reranker 的实现。
"""
import threading
import time

import pytest

import modular_rag_mcp_server.libs.reranker.cross_encoder_reranker as cross_encoder_module
from modular_rag_mcp_server.core.settings import RerankConfig
from modular_rag_mcp_server.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from modular_rag_mcp_server.libs.vector_store.base_vector_store import QueryResult


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    """使用有界轮询等待异步条件成立。"""
    deadline = time.monotonic() + timeout
    poll_interval = threading.Event()
    while not predicate():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        poll_interval.wait(min(0.01, remaining))
    return True


def _mock_scorer(query: str, text: str) -> float:
    """
    Mock scorer 函数（用于测试）

    简单的确定性打分函数：基于文本长度和查询匹配度
    """
    # 简单的打分逻辑：文本长度越短，分数越高（模拟相关性）
    base_score = 1.0 / (len(text) + 1)

    # 如果查询在文本中，增加分数
    if query.lower() in text.lower():
        base_score += 0.5

    return base_score


def test_factory_creates_cross_encoder_reranker():
    """测试工厂可以创建 Cross-Encoder Reranker（使用 mock scorer）"""
    rerank_config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    # 使用 mock scorer 创建 reranker
    reranker = CrossEncoderReranker(rerank_config, scorer=_mock_scorer)

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.get_backend() == "cross_encoder"


def test_cross_encoder_reranker_initialization_with_mock():
    """测试 Cross-Encoder Reranker 使用 mock scorer 初始化"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(config, scorer=_mock_scorer)

    assert reranker.get_backend() == "cross_encoder"
    assert reranker._use_mock is True
    assert reranker._scorer == _mock_scorer


def test_cross_encoder_reranker_missing_model():
    """测试模型名称为空时抛出错误"""
    config = RerankConfig(
        backend="cross_encoder",
        model="",
        top_m=30,
        timeout_seconds=5
    )

    with pytest.raises(ValueError, match="模型名称不能为空"):
        CrossEncoderReranker(config, scorer=_mock_scorer)


def test_cross_encoder_reranker_initialization_without_scorer(monkeypatch):
    """不传 scorer 时初始化 CrossEncoder 依赖，不访问网络或模型缓存。"""
    created_models = []

    class FakeCrossEncoder:
        def __init__(self, model_name):
            created_models.append(model_name)

    monkeypatch.setattr(
        cross_encoder_module,
        "CROSS_ENCODER_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        cross_encoder_module,
        "CrossEncoder",
        FakeCrossEncoder,
    )
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(config)

    assert reranker.get_backend() == "cross_encoder"
    assert reranker._use_mock is False
    assert reranker._model is not None
    assert created_models == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]


def test_cross_encoder_reranker_not_installed(monkeypatch):
    """测试 CrossEncoder 未安装时抛出错误"""
    monkeypatch.setattr(
        cross_encoder_module,
        "CROSS_ENCODER_AVAILABLE",
        False,
    )
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    with pytest.raises(RuntimeError, match="CrossEncoder 未安装"):
        CrossEncoderReranker(config)


def test_cross_encoder_reranker_empty_query():
    """测试查询为空时抛出错误"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(config, scorer=_mock_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2")
    ]

    with pytest.raises(ValueError, match="查询文本不能为空"):
        reranker.rerank("", candidates)


def test_cross_encoder_reranker_empty_candidates():
    """测试候选列表为空时抛出错误"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(config, scorer=_mock_scorer)

    with pytest.raises(ValueError, match="候选列表不能为空"):
        reranker.rerank("test query", [])


def test_cross_encoder_reranker_single_candidate():
    """测试单个候选时直接返回"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(config, scorer=_mock_scorer)
    candidates = [QueryResult(id="id1", score=0.9, text="text1")]

    result = reranker.rerank("test query", candidates)

    assert len(result) == 1
    assert result[0].id == "id1"


def test_cross_encoder_reranker_successful_rerank():
    """测试成功重排序"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(config, scorer=_mock_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="short text"),  # 应该得分最高
        QueryResult(id="id2", score=0.8, text="this is a much longer text that should score lower"),
        QueryResult(id="id3", score=0.7, text="medium length text")
    ]

    result = reranker.rerank("test query", candidates)

    assert len(result) == 3
    # 由于 mock scorer 的逻辑，较短的文本应该得分更高
    assert result[0].id == "id1"  # "short text" 最短
    # 验证分数已更新
    assert result[0].score > 0
    assert result[0].score != candidates[0].score  # 分数应该被更新


def test_cross_encoder_reranker_top_m_limit():
    """测试 top_m 限制功能"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=2,  # 只处理前 2 个
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(config, scorer=_mock_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2"),
        QueryResult(id="id3", score=0.7, text="text3"),
        QueryResult(id="id4", score=0.6, text="text4")
    ]

    result = reranker.rerank("test query", candidates)

    # 应该只返回 top_m=2 个结果
    assert len(result) == 2
    assert result[0].id in ["id1", "id2"]
    assert result[1].id in ["id1", "id2"]


def test_cross_encoder_reranker_timeout():
    """超时必须在截止时间附近返回，而不是等待后台评分完成。"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=0.05,
    )

    # 创建一个会延迟的 mock scorer
    def slow_scorer(query: str, text: str) -> float:
        time.sleep(0.3)
        return 0.5

    reranker = CrossEncoderReranker(config, scorer=slow_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2")
    ]

    started_at = time.monotonic()
    with pytest.raises(RuntimeError, match="超时"):
        reranker.rerank("test query", candidates)
    assert time.monotonic() - started_at < 0.2


def test_cross_encoder_rejects_new_work_while_timed_out_task_runs():
    """一个超时任务仍运行时不得再启动新的评分线程。"""
    release = threading.Event()
    started = threading.Event()
    scorer_calls = 0

    def blocking_scorer(query: str, text: str) -> float:
        nonlocal scorer_calls
        scorer_calls += 1
        started.set()
        release.wait(timeout=1)
        return 0.5

    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=0.05,
    )
    reranker = CrossEncoderReranker(config, scorer=blocking_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2"),
    ]

    try:
        with pytest.raises(RuntimeError, match="超时"):
            reranker.rerank("test query", candidates)
        assert started.is_set()

        started_at = time.monotonic()
        with pytest.raises(RuntimeError, match="仍在运行"):
            reranker.rerank("test query", candidates)
        assert time.monotonic() - started_at < 0.05
        assert scorer_calls == 1
    finally:
        release.set()


def test_cross_encoder_recovers_after_timed_out_task_finishes():
    """超时后台任务结束后，同一实例必须能够接受新的评分任务。"""
    release = threading.Event()
    started = threading.Event()
    scorer_calls = 0
    scorer_calls_lock = threading.Lock()

    def blocking_scorer(query: str, text: str) -> float:
        nonlocal scorer_calls
        started.set()
        if not release.wait(timeout=1):
            raise RuntimeError("测试未及时释放 scorer")
        with scorer_calls_lock:
            scorer_calls += 1
        return 0.5

    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=0.05,
    )
    reranker = CrossEncoderReranker(config, scorer=blocking_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2"),
    ]

    try:
        with pytest.raises(RuntimeError, match="超时"):
            reranker.rerank("test query", candidates)
        assert started.is_set()

        timed_out_future = reranker._active_future
        assert timed_out_future is not None
        release.set()
        assert _wait_until(
            lambda: (
                timed_out_future.done()
                and reranker._active_future is None
                and reranker._active_executor is None
            )
        )

        result = reranker.rerank("test query", candidates)

        assert len(result) == 2
        assert scorer_calls == 4
    finally:
        release.set()


def test_cross_encoder_preserves_scorer_timeout_error():
    """scorer 自身的 TimeoutError 必须按普通调用失败处理。"""
    scorer_timeout = TimeoutError("scorer internal timeout")

    def timeout_scorer(query: str, text: str) -> float:
        raise scorer_timeout

    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=1,
    )
    reranker = CrossEncoderReranker(config, scorer=timeout_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2"),
    ]

    with pytest.raises(RuntimeError, match="调用失败") as exc_info:
        reranker.rerank("test query", candidates)

    assert exc_info.value.__cause__ is scorer_timeout


def test_cross_encoder_submit_cleans_up_and_reraises_base_exception(monkeypatch):
    """submit 的 BaseException 必须清理 executor 后原样传播。"""
    shutdown_called = False
    submit_interrupt = KeyboardInterrupt("submit interrupted")

    class InterruptingExecutor:
        def __init__(self, max_workers: int):
            assert max_workers == 1

        def submit(self, function):
            raise submit_interrupt

        def shutdown(self, wait: bool, cancel_futures: bool = False):
            nonlocal shutdown_called
            shutdown_called = True

    monkeypatch.setattr(
        "modular_rag_mcp_server.libs.reranker.cross_encoder_reranker.ThreadPoolExecutor",
        InterruptingExecutor,
    )
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=1,
    )
    reranker = CrossEncoderReranker(config, scorer=_mock_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2"),
    ]

    with pytest.raises(KeyboardInterrupt, match="submit interrupted") as exc_info:
        reranker.rerank("test query", candidates)

    assert exc_info.value is submit_interrupt
    assert shutdown_called is True


def test_cross_encoder_recovers_after_scorer_error():
    """评分异常清理完成后，同一实例的下一次调用必须成功。"""
    scorer_calls = 0

    def flaky_scorer(query: str, text: str) -> float:
        nonlocal scorer_calls
        scorer_calls += 1
        if scorer_calls == 1:
            raise ValueError("模拟评分异常")
        return 0.5

    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=1,
    )
    reranker = CrossEncoderReranker(config, scorer=flaky_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2"),
    ]

    with pytest.raises(RuntimeError, match="模拟评分异常"):
        reranker.rerank("test query", candidates)

    result = reranker.rerank("test query", candidates)

    assert len(result) == 2
    assert scorer_calls == 3


def test_cross_encoder_allows_repeated_successful_calls():
    """正常评分完成后必须及时清理，以支持同一实例重复调用。"""
    scorer_calls = 0

    def counting_scorer(query: str, text: str) -> float:
        nonlocal scorer_calls
        scorer_calls += 1
        return 0.5

    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=1,
    )
    reranker = CrossEncoderReranker(config, scorer=counting_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2"),
    ]

    first_result = reranker.rerank("first query", candidates)
    second_result = reranker.rerank("second query", candidates)

    assert len(first_result) == 2
    assert len(second_result) == 2
    assert scorer_calls == 4


def test_cross_encoder_reranker_score_ordering():
    """测试分数排序正确性"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    # 创建一个确定性 scorer，返回预定义的分数
    def deterministic_scorer(query: str, text: str) -> float:
        score_map = {
            "text1": 0.9,
            "text2": 0.7,
            "text3": 0.8
        }
        return score_map.get(text, 0.5)

    reranker = CrossEncoderReranker(config, scorer=deterministic_scorer)
    candidates = [
        QueryResult(id="id1", score=0.5, text="text1"),
        QueryResult(id="id2", score=0.5, text="text2"),
        QueryResult(id="id3", score=0.5, text="text3")
    ]

    result = reranker.rerank("test query", candidates)

    # 验证排序：text1 (0.9) > text3 (0.8) > text2 (0.7)
    assert len(result) == 3
    assert result[0].id == "id1"  # 最高分
    assert result[0].score == 0.9
    assert result[1].id == "id3"  # 第二高分
    assert result[1].score == 0.8
    assert result[2].id == "id2"  # 最低分
    assert result[2].score == 0.7


def test_cross_encoder_reranker_metadata_preserved():
    """测试元数据被保留"""
    config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(config, scorer=_mock_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1", metadata={"key1": "value1"}),
        QueryResult(id="id2", score=0.8, text="text2", metadata={"key2": "value2"})
    ]

    result = reranker.rerank("test query", candidates)

    assert len(result) == 2
    # 验证元数据被保留
    assert result[0].metadata == {"key1": "value1"} or result[0].metadata == {"key2": "value2"}
    assert result[1].metadata == {"key1": "value1"} or result[1].metadata == {"key2": "value2"}


def test_cross_encoder_reranker_factory_integration():
    """测试工厂集成（使用 mock scorer）"""
    rerank_config = RerankConfig(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m=30,
        timeout_seconds=5
    )

    reranker = CrossEncoderReranker(rerank_config, scorer=_mock_scorer)
    candidates = [
        QueryResult(id="id1", score=0.9, text="text1"),
        QueryResult(id="id2", score=0.8, text="text2")
    ]

    result = reranker.rerank("test query", candidates)
    assert len(result) == 2
