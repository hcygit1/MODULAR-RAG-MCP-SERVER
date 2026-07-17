"""
TraceContext — 请求级链路追踪上下文

每次检索或摄取请求创建一个 TraceContext 实例，通过 record_stage() 记录
各阶段（dense、sparse、fusion、rerank 等）的耗时与指标，finish() 生成
可 JSON 序列化的完整 trace 字典。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageRecord:
    """单个阶段的追踪记录"""

    name: str
    start_ms: float
    end_ms: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class TraceContext:
    """
    请求级链路追踪上下文

    用法::

        trace = TraceContext(operation="retrieval")
        with trace.stage("dense_retrieval", top_k=20):
            results = dense.retrieve(...)
        with trace.stage("rerank"):
            ranked = reranker.rerank(...)
        trace.set_metric("total_results", len(ranked))
        report = trace.finish()

    也可以手动记录::

        trace.record_stage("fusion", duration_ms=12.5, result_count=30)
    """

    def __init__(
        self,
        operation: str = "unknown",
        trace_id: Optional[str] = None,
    ) -> None:
        self.trace_id: str = trace_id or uuid.uuid4().hex[:16]
        self.operation: str = operation
        self._created_at: float = time.monotonic()
        self._created_ts: float = time.time()
        self._stages: List[StageRecord] = []
        self._metrics: Dict[str, Any] = {}
        self._finished: bool = False

    @property
    def stages(self) -> List[StageRecord]:
        return list(self._stages)

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    def record_stage(
        self,
        name: str,
        duration_ms: Optional[float] = None,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None,
        **metadata: Any,
    ) -> StageRecord:
        """
        手动追加一个阶段记录。

        可以直接提供 duration_ms，也可提供 start_ms/end_ms（均为相对 trace 创建时间的偏移毫秒数）。
        未提供时间参数时，记录当前时刻作为零长度阶段。
        """
        now_offset = (time.monotonic() - self._created_at) * 1000.0

        if duration_ms is not None and start_ms is None and end_ms is None:
            _end = now_offset
            _start = _end - duration_ms
            _dur = duration_ms
        elif start_ms is not None and end_ms is not None:
            _start = start_ms
            _end = end_ms
            _dur = _end - _start
        else:
            _start = now_offset
            _end = now_offset
            _dur = 0.0

        rec = StageRecord(
            name=name,
            start_ms=_start,
            end_ms=_end,
            duration_ms=_dur,
            metadata=dict(metadata) if metadata else {},
        )
        self._stages.append(rec)
        return rec

    def stage(self, name: str, **metadata: Any) -> _StageTimer:
        """
        返回上下文管理器，自动测量阶段耗时。

        用法::

            with trace.stage("dense_retrieval", top_k=20):
                results = dense.retrieve(...)
        """
        return _StageTimer(self, name, metadata)

    def set_metric(self, key: str, value: Any) -> None:
        """设置顶层指标（如 total_results、query 等）"""
        self._metrics[key] = value

    def finish(self) -> Dict[str, Any]:
        """
        结束追踪，返回可 JSON 序列化的完整 trace 字典。

        多次调用 finish() 是安全的，但 total_duration_ms 仅以首次 finish 时间计算。
        """
        if not self._finished:
            self._finished = True
            self._finish_offset = (time.monotonic() - self._created_at) * 1000.0

        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "timestamp": self._created_ts,
            "total_duration_ms": round(self._finish_offset, 3),
            "stages": [s.to_dict() for s in self._stages],
            "metrics": self._metrics,
        }


class _StageTimer:
    """stage() 返回的上下文管理器，自动测量阶段耗时"""

    def __init__(
        self,
        ctx: TraceContext,
        name: str,
        metadata: Dict[str, Any],
    ) -> None:
        self._ctx = ctx
        self._name = name
        self._metadata = metadata
        self._start: float = 0.0

    def __enter__(self) -> _StageTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_ms = (time.monotonic() - self._start) * 1000.0
        start_offset = (self._start - self._ctx._created_at) * 1000.0
        end_offset = start_offset + elapsed_ms

        meta = dict(self._metadata)
        if exc_type is not None:
            meta["error"] = str(exc_val)

        self._ctx._stages.append(
            StageRecord(
                name=self._name,
                start_ms=start_offset,
                end_ms=end_offset,
                duration_ms=elapsed_ms,
                metadata=meta,
            )
        )
