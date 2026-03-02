"""
Evaluation Runner — 读取黄金测试集，运行 Retrieval 并产出评估指标

核心职责：
1. 加载 golden_test_set.json（query + golden_chunk_ids）
2. 对每条 query 执行 retrieval，收集 retrieved_ids
3. 调用 Evaluator(s) 计算 hit_rate / mrr 等指标
4. 汇总输出 per-query 明细与总体均值
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

logger = logging.getLogger(__name__)


@dataclass
class GoldenCase:
    """一条黄金测试用例"""

    query: str
    golden_chunk_ids: List[str]
    collection: Optional[str] = None
    top_k: int = 10
    description: str = ""


@dataclass
class CaseResult:
    """单条用例的评估结果"""

    query: str
    description: str
    golden_chunk_ids: List[str]
    retrieved_ids: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalReport:
    """完整评估报告"""

    total_cases: int
    successful_cases: int
    failed_cases: int
    avg_metrics: Dict[str, float]
    case_results: List[CaseResult]
    total_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_cases": self.total_cases,
                "successful_cases": self.successful_cases,
                "failed_cases": self.failed_cases,
                "avg_metrics": {k: round(v, 4) for k, v in self.avg_metrics.items()},
                "total_time_ms": round(self.total_time_ms, 2),
            },
            "case_results": [
                {
                    "query": r.query,
                    "description": r.description,
                    "golden_chunk_ids": r.golden_chunk_ids,
                    "retrieved_ids": r.retrieved_ids,
                    "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                }
                for r in self.case_results
            ],
        }


# 检索回调函数签名：(query, top_k, collection) -> List[str]
RetrieveFunc = Callable[[str, int, Optional[str]], List[str]]


def load_golden_test_set(path: str) -> List[GoldenCase]:
    """
    加载黄金测试集

    Args:
        path: golden_test_set.json 文件路径

    Returns:
        解析后的测试用例列表

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: JSON 格式不合法或缺少必填字段
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"黄金测试集文件不存在: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"黄金测试集必须是 JSON 数组，得到: {type(data).__name__}")

    cases: List[GoldenCase] = []
    for i, item in enumerate(data):
        if "query" not in item:
            raise ValueError(f"第 {i} 条测试用例缺少 'query' 字段")
        if "golden_chunk_ids" not in item:
            raise ValueError(f"第 {i} 条测试用例缺少 'golden_chunk_ids' 字段")

        cases.append(
            GoldenCase(
                query=item["query"],
                golden_chunk_ids=item["golden_chunk_ids"],
                collection=item.get("collection"),
                top_k=item.get("top_k", 10),
                description=item.get("description", ""),
            )
        )

    return cases


class EvalRunner:
    """
    评估执行器

    接收一个检索回调函数和评估器列表，遍历黄金测试集执行评估。

    用法::

        def my_retrieve(query, top_k, collection):
            results = pipeline.retrieve(query, top_k, collection)
            return [r.id for r in results]

        runner = EvalRunner(retrieve_func=my_retrieve)
        report = runner.run("tests/fixtures/golden_test_set.json")
        print(report.to_dict())
    """

    def __init__(
        self,
        retrieve_func: RetrieveFunc,
        evaluators: Optional[List[BaseEvaluator]] = None,
    ) -> None:
        """
        Args:
            retrieve_func: 检索回调，签名 (query, top_k, collection) -> List[str]，
                           返回检索到的 chunk_id 列表（按相关性排序）
            evaluators: 评估器列表，默认使用 [CustomEvaluator()]
        """
        self._retrieve = retrieve_func
        self._evaluators = evaluators or [CustomEvaluator()]

    def run(self, golden_set_path: str) -> EvalReport:
        """
        执行完整评估流程

        Args:
            golden_set_path: 黄金测试集 JSON 文件路径

        Returns:
            EvalReport: 评估报告，包含 per-query 明细和总体均值
        """
        cases = load_golden_test_set(golden_set_path)
        logger.info("加载黄金测试集: %d 条用例", len(cases))

        case_results: List[CaseResult] = []
        run_start = time.monotonic()

        for i, case in enumerate(cases):
            logger.info(
                "[%d/%d] 评估: %s",
                i + 1,
                len(cases),
                case.query[:50],
            )
            result = self._evaluate_case(case)
            case_results.append(result)

        total_time_ms = (time.monotonic() - run_start) * 1000.0

        successful = [r for r in case_results if r.error is None]
        failed = [r for r in case_results if r.error is not None]

        avg_metrics = self._aggregate_metrics(successful)

        report = EvalReport(
            total_cases=len(cases),
            successful_cases=len(successful),
            failed_cases=len(failed),
            avg_metrics=avg_metrics,
            case_results=case_results,
            total_time_ms=total_time_ms,
        )

        logger.info(
            "评估完成: %d/%d 成功, 平均指标: %s",
            len(successful),
            len(cases),
            {k: f"{v:.4f}" for k, v in avg_metrics.items()},
        )

        return report

    def _evaluate_case(self, case: GoldenCase) -> CaseResult:
        """评估单条测试用例"""
        result = CaseResult(
            query=case.query,
            description=case.description,
            golden_chunk_ids=case.golden_chunk_ids,
            retrieved_ids=[],
        )

        try:
            start = time.monotonic()
            retrieved_ids = self._retrieve(
                case.query,
                case.top_k,
                case.collection,
            )
            result.latency_ms = (time.monotonic() - start) * 1000.0
            result.retrieved_ids = retrieved_ids

            if not retrieved_ids:
                for evaluator in self._evaluators:
                    prefix = evaluator.get_backend()
                    result.metrics[f"{prefix}_hit_rate"] = 0.0
                    result.metrics[f"{prefix}_mrr"] = 0.0
                return result

            merged: Dict[str, float] = {}
            for evaluator in self._evaluators:
                metrics = evaluator.evaluate(
                    query=case.query,
                    retrieved_ids=retrieved_ids,
                    golden_ids=case.golden_chunk_ids,
                )
                for key, val in metrics.items():
                    prefix = evaluator.get_backend()
                    merged[f"{prefix}_{key}"] = val
            result.metrics = merged

        except Exception as exc:
            logger.warning("用例评估失败 [%s]: %s", case.query[:30], exc)
            result.error = str(exc)

        return result

    def _aggregate_metrics(
        self, results: List[CaseResult]
    ) -> Dict[str, float]:
        """对成功用例的指标取算术均值"""
        if not results:
            return {}

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for r in results:
            for key, val in r.metrics.items():
                sums[key] = sums.get(key, 0.0) + val
                counts[key] = counts.get(key, 0) + 1

        return {k: sums[k] / counts[k] for k in sums}
