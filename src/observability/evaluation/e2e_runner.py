"""
E2E 评估编排器

串联 retrieve → build_mcp_content，执行 L1 检索评估 + L2 内容评估。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.libs.vector_store.base_vector_store import QueryResult

from src.observability.evaluation.content_evaluator import ContentEvaluator, evaluate_content
from src.observability.evaluation.eval_runner import (
    CaseResult,
    EvalReport,
    GoldenCase,
    load_golden_test_set,
)
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

logger = logging.getLogger(__name__)

# RAG 回调：(query, top_k, collection) -> (results: List[QueryResult], mcp_content: Dict)
RagFunc = Callable[
    [str, int, Optional[str]],
    Tuple[List[QueryResult], Dict[str, Any]],
]


class E2ERunner:
    """
    E2E 评估执行器

    接收 RAG 回调（retrieve + build_mcp_content），对每条用例执行 L1 + L2 评估。
    """

    def __init__(
        self,
        rag_func: RagFunc,
        l1_evaluators: Optional[List[BaseEvaluator]] = None,
    ) -> None:
        """
        Args:
            rag_func: (query, top_k, collection) -> (results, mcp_content)
            l1_evaluators: L1 检索评估器，默认 [CustomEvaluator()]
        """
        self._rag = rag_func
        self._l1_evaluators = l1_evaluators or [CustomEvaluator()]
        self._content_evaluator = ContentEvaluator()

    def run(self, golden_set_path: str) -> EvalReport:
        """
        执行 E2E 评估流程

        Args:
            golden_set_path: 黄金测试集 JSON 文件路径

        Returns:
            EvalReport: 含 L1 + L2 指标的评估报告
        """
        cases = load_golden_test_set(golden_set_path)
        logger.info("加载黄金测试集: %d 条用例 (E2E 模式)", len(cases))

        case_results: List[CaseResult] = []
        run_start = time.monotonic()

        for i, case in enumerate(cases):
            logger.info("[%d/%d] E2E 评估: %s", i + 1, len(cases), case.query[:50])
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
            "E2E 评估完成: %d/%d 成功, 平均指标: %s",
            len(successful),
            len(cases),
            {k: f"{v:.4f}" for k, v in avg_metrics.items()},
        )

        return report

    def _evaluate_case(self, case: GoldenCase) -> CaseResult:
        """评估单条用例：执行 RAG → L1 + L2"""
        result = CaseResult(
            query=case.query,
            description=case.description,
            golden_chunk_ids=case.golden_chunk_ids,
            retrieved_ids=[],
        )

        try:
            start = time.monotonic()
            results, mcp_content = self._rag(
                case.query,
                case.top_k,
                case.collection,
            )
            result.latency_ms = (time.monotonic() - start) * 1000.0
            result.retrieved_ids = [r.id for r in results]

            # L1 检索评估
            if not result.retrieved_ids:
                for ev in self._l1_evaluators:
                    prefix = ev.get_backend()
                    result.metrics[f"{prefix}_hit_rate"] = 0.0
                    result.metrics[f"{prefix}_mrr"] = 0.0
            else:
                for ev in self._l1_evaluators:
                    l1_metrics = ev.evaluate(
                        query=case.query,
                        retrieved_ids=result.retrieved_ids,
                        golden_ids=case.golden_chunk_ids,
                    )
                    for key, val in l1_metrics.items():
                        result.metrics[f"{ev.get_backend()}_{key}"] = val

            # L2 内容评估
            l2_metrics = evaluate_content(
                mcp_content,
                case.expected_content_checks,
            )
            for key, val in l2_metrics.items():
                result.metrics[f"content_{key}"] = val

        except Exception as exc:
            logger.warning("E2E 用例评估失败 [%s]: %s", case.query[:30], exc)
            result.error = str(exc)

        return result

    def _aggregate_metrics(self, results: List[CaseResult]) -> Dict[str, float]:
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
