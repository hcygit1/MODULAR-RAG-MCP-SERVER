"""
Reranker Fallback 单元测试

模拟后端异常时，验证不影响最终返回，且标记 fallback=true。
"""
from unittest.mock import MagicMock

import pytest

from src.core.query_engine.reranker import RerankerOrchestrator
from src.libs.vector_store.base_vector_store import QueryResult


@pytest.fixture
def sample_candidates():
    """测试用候选列表"""
    return [
        QueryResult("c1", 0.8, "text one", {"source": "a.pdf"}),
        QueryResult("c2", 0.6, "text two", {"source": "a.pdf"}),
        QueryResult("c3", 0.4, "text three", {"source": "b.pdf"}),
    ]


class TestRerankerFallbackBasic:
    """基础功能测试"""

    def test_backend_success_returns_reranked_no_fallback(
        self,
        sample_candidates,
    ) -> None:
        """后端正常时返回精排结果，fallback=False"""
        mock_backend = MagicMock()
        mock_backend.rerank.return_value = [
            QueryResult("c2", 0.95, "text two", {"source": "a.pdf"}),
            QueryResult("c1", 0.7, "text one", {"source": "a.pdf"}),
            QueryResult("c3", 0.3, "text three", {"source": "b.pdf"}),
        ]

        orch = RerankerOrchestrator(backend=mock_backend)
        results, fallback = orch.rerank_with_fallback("query", sample_candidates)

        assert fallback is False
        assert len(results) == 3
        assert results[0].id == "c2"
        mock_backend.rerank.assert_called_once()

    def test_backend_exception_returns_original_fallback_true(
        self,
        sample_candidates,
    ) -> None:
        """后端异常时返回原始 fusion 排名，fallback=True"""
        mock_backend = MagicMock()
        mock_backend.rerank.side_effect = RuntimeError("Model loading failed")

        orch = RerankerOrchestrator(backend=mock_backend)
        results, fallback = orch.rerank_with_fallback("query", sample_candidates)

        assert fallback is True
        assert len(results) == 3
        assert [r.id for r in results] == ["c1", "c2", "c3"]
        assert results[0].score == 0.8
        assert results[0].text == "text one"

    def test_backend_value_error_fallback(
        self,
        sample_candidates,
    ) -> None:
        """后端抛出 ValueError 时也回退"""
        mock_backend = MagicMock()
        mock_backend.rerank.side_effect = ValueError("Empty candidates")

        orch = RerankerOrchestrator(backend=mock_backend)
        results, fallback = orch.rerank_with_fallback("query", sample_candidates)

        assert fallback is True
        assert [r.id for r in results] == ["c1", "c2", "c3"]

    def test_empty_candidates_returns_empty_no_fallback(self) -> None:
        """空候选返回空列表，fallback=False"""
        mock_backend = MagicMock()

        orch = RerankerOrchestrator(backend=mock_backend)
        results, fallback = orch.rerank_with_fallback("query", [])

        assert fallback is False
        assert results == []
        mock_backend.rerank.assert_not_called()

    def test_fallback_returns_copy_not_mutation(
        self,
        sample_candidates,
    ) -> None:
        """fallback 时返回的是副本，修改结果不影响原 candidates"""
        mock_backend = MagicMock()
        mock_backend.rerank.side_effect = RuntimeError("oops")

        orch = RerankerOrchestrator(backend=mock_backend)
        results, _ = orch.rerank_with_fallback("query", sample_candidates)

        results[0].metadata["x"] = "modified"
        assert sample_candidates[0].metadata.get("x") != "modified"
