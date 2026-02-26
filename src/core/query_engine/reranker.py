"""
Reranker 编排模块

接入 libs.reranker 后端，对 fusion 结果进行精排；失败或超时时回退到 fusion 排名并标记 fallback。
"""
from typing import Any, List, Optional, Tuple

from src.core.trace.trace_context import TraceContext
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.vector_store.base_vector_store import QueryResult


class RerankerOrchestrator:
    """
    Reranker 编排器

    调用 libs.reranker 后端进行精排，捕获异常/超时后回退到原始 fusion 排名，
    并标记 fallback 供上层感知。
    """

    def __init__(self, backend: BaseReranker) -> None:
        self._backend = backend

    def rerank_with_fallback(
        self,
        query: str,
        candidates: List[QueryResult],
        trace: Optional[Any] = None,
    ) -> Tuple[List[QueryResult], bool]:
        """
        执行精排，失败/超时时回退到 fusion 排名

        Args:
            query: 查询文本
            candidates: fusion 融合后的候选列表
            trace: 追踪上下文（可选）

        Returns:
            Tuple[List[QueryResult], bool]: (精排后的结果, 是否发生 fallback)
        """
        if not candidates:
            return ([], False)

        _trace: Optional[TraceContext] = trace if isinstance(trace, TraceContext) else None

        try:
            if _trace:
                with _trace.stage("rerank", backend=self._backend.get_backend(), candidate_count=len(candidates)):
                    results = self._backend.rerank(query=query, candidates=candidates, trace=trace)
                _trace.stages[-1].metadata["fallback"] = False
            else:
                results = self._backend.rerank(query=query, candidates=candidates, trace=trace)
            return (results, False)
        except Exception:
            if _trace:
                _trace.record_stage(
                    "rerank",
                    duration_ms=0.0,
                    backend=self._backend.get_backend(),
                    candidate_count=len(candidates),
                    fallback=True,
                )
            return (self._copy_candidates(candidates), True)

    def _copy_candidates(self, candidates: List[QueryResult]) -> List[QueryResult]:
        """复制 candidates 列表（保持顺序，不修改原列表）"""
        return [
            QueryResult(
                id=c.id,
                score=c.score,
                text=c.text,
                metadata=c.metadata.copy() if c.metadata else {},
            )
            for c in candidates
        ]
