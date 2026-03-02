"""
EvalRunner 单元测试

验证 eval_runner 的核心流程：加载测试集、调用检索、计算指标、聚合报告。
"""
import json
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest

from src.observability.evaluation.eval_runner import (
    CaseResult,
    EvalReport,
    EvalRunner,
    GoldenCase,
    load_golden_test_set,
)


# ── 辅助工厂 ──────────────────────────────────────────────────────


def _write_golden_set(cases: list, path: str) -> None:
    """将测试用例写入 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)


def _perfect_retrieve(query: str, top_k: int, collection: Optional[str]) -> List[str]:
    """模拟完美命中：golden ID 始终排在第 1 位"""
    if "q1" in query:
        return ["chunk_1", "chunk_2", "chunk_3"]
    if "q2" in query:
        return ["chunk_2", "chunk_1", "chunk_3"]
    return ["chunk_1", "chunk_2", "chunk_3"]


def _partial_retrieve(query: str, top_k: int, collection: Optional[str]) -> List[str]:
    """模拟部分命中：第 2 位命中"""
    return ["miss_1", "chunk_1", "miss_2"]


def _empty_retrieve(query: str, top_k: int, collection: Optional[str]) -> List[str]:
    """模拟空结果"""
    return []


def _failing_retrieve(query: str, top_k: int, collection: Optional[str]) -> List[str]:
    """模拟检索异常"""
    raise RuntimeError("检索服务不可用")


# ── load_golden_test_set 测试 ─────────────────────────────────────


class TestLoadGoldenTestSet:

    def test_load_valid_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "golden.json")
        _write_golden_set(
            [
                {"query": "q1", "golden_chunk_ids": ["c1"], "top_k": 5},
                {"query": "q2", "golden_chunk_ids": ["c2", "c3"]},
            ],
            path,
        )

        cases = load_golden_test_set(path)
        assert len(cases) == 2
        assert cases[0].query == "q1"
        assert cases[0].golden_chunk_ids == ["c1"]
        assert cases[0].top_k == 5
        assert cases[1].top_k == 10  # 默认值

    def test_load_with_optional_fields(self, tmp_path: Path) -> None:
        path = str(tmp_path / "golden.json")
        _write_golden_set(
            [
                {
                    "query": "q1",
                    "golden_chunk_ids": ["c1"],
                    "collection": "my_coll",
                    "description": "测试说明",
                }
            ],
            path,
        )

        cases = load_golden_test_set(path)
        assert cases[0].collection == "my_coll"
        assert cases[0].description == "测试说明"

    def test_load_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="不存在"):
            load_golden_test_set("/nonexistent/golden.json")

    def test_load_invalid_format_not_array(self, tmp_path: Path) -> None:
        path = str(tmp_path / "golden.json")
        with open(path, "w") as f:
            json.dump({"query": "q"}, f)

        with pytest.raises(ValueError, match="JSON 数组"):
            load_golden_test_set(path)

    def test_load_missing_query(self, tmp_path: Path) -> None:
        path = str(tmp_path / "golden.json")
        _write_golden_set([{"golden_chunk_ids": ["c1"]}], path)

        with pytest.raises(ValueError, match="query"):
            load_golden_test_set(path)

    def test_load_missing_golden_ids(self, tmp_path: Path) -> None:
        path = str(tmp_path / "golden.json")
        _write_golden_set([{"query": "q1"}], path)

        with pytest.raises(ValueError, match="golden_chunk_ids"):
            load_golden_test_set(path)


# ── EvalRunner 测试 ───────────────────────────────────────────────


class TestEvalRunner:

    @pytest.fixture
    def golden_path(self, tmp_path: Path) -> str:
        path = str(tmp_path / "golden.json")
        _write_golden_set(
            [
                {"query": "q1", "golden_chunk_ids": ["chunk_1"]},
                {"query": "q2", "golden_chunk_ids": ["chunk_2"]},
            ],
            path,
        )
        return path

    def test_perfect_retrieval(self, golden_path: str) -> None:
        """完美命中：hit_rate=1.0, mrr=1.0"""
        runner = EvalRunner(retrieve_func=_perfect_retrieve)
        report = runner.run(golden_path)

        assert report.total_cases == 2
        assert report.successful_cases == 2
        assert report.failed_cases == 0
        assert report.avg_metrics["custom_hit_rate"] == 1.0
        assert report.avg_metrics["custom_mrr"] == 1.0

    def test_partial_retrieval(self, tmp_path: Path) -> None:
        """部分命中：chunk_1 在第 2 位"""
        path = str(tmp_path / "golden.json")
        _write_golden_set(
            [{"query": "q1", "golden_chunk_ids": ["chunk_1"]}],
            path,
        )

        runner = EvalRunner(retrieve_func=_partial_retrieve)
        report = runner.run(path)

        assert report.successful_cases == 1
        assert report.avg_metrics["custom_hit_rate"] == 1.0
        assert report.avg_metrics["custom_mrr"] == 0.5  # 1/2

    def test_empty_retrieval(self, tmp_path: Path) -> None:
        """空结果：hit_rate=0, mrr=0"""
        path = str(tmp_path / "golden.json")
        _write_golden_set(
            [{"query": "q1", "golden_chunk_ids": ["chunk_1"]}],
            path,
        )

        runner = EvalRunner(retrieve_func=_empty_retrieve)
        report = runner.run(path)

        assert report.successful_cases == 1
        assert report.avg_metrics["custom_hit_rate"] == 0.0
        assert report.avg_metrics["custom_mrr"] == 0.0

    def test_failing_retrieval_marks_error(self, golden_path: str) -> None:
        """检索失败时用例标记 error，不中断其他用例"""
        runner = EvalRunner(retrieve_func=_failing_retrieve)
        report = runner.run(golden_path)

        assert report.total_cases == 2
        assert report.failed_cases == 2
        for r in report.case_results:
            assert r.error is not None
            assert "不可用" in r.error

    def test_mixed_success_and_failure(self, tmp_path: Path) -> None:
        """混合成功/失败用例，均值只算成功的"""
        path = str(tmp_path / "golden.json")
        _write_golden_set(
            [
                {"query": "ok", "golden_chunk_ids": ["chunk_1"]},
                {"query": "fail", "golden_chunk_ids": ["chunk_x"]},
            ],
            path,
        )

        call_count = 0

        def mixed_retrieve(q: str, top_k: int, coll: Optional[str]) -> List[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ["chunk_1"]
            raise RuntimeError("boom")

        runner = EvalRunner(retrieve_func=mixed_retrieve)
        report = runner.run(path)

        assert report.successful_cases == 1
        assert report.failed_cases == 1
        assert report.avg_metrics["custom_hit_rate"] == 1.0

    def test_report_to_dict(self, golden_path: str) -> None:
        """to_dict() 产出可 JSON 序列化的结构"""
        runner = EvalRunner(retrieve_func=_perfect_retrieve)
        report = runner.run(golden_path)
        d = report.to_dict()

        serialized = json.dumps(d, ensure_ascii=False)
        parsed = json.loads(serialized)

        assert "summary" in parsed
        assert "case_results" in parsed
        assert parsed["summary"]["total_cases"] == 2

    def test_latency_tracked(self, golden_path: str) -> None:
        """检索延迟被记录"""
        runner = EvalRunner(retrieve_func=_perfect_retrieve)
        report = runner.run(golden_path)

        for r in report.case_results:
            assert r.latency_ms >= 0.0
        assert report.total_time_ms >= 0.0
