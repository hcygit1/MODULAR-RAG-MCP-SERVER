"""
TraceCollector — 收集并持久化 trace 数据

负责将 TraceContext.finish() 产出的 dict 写入 JSON Lines 文件，
供后续 Dashboard / Evaluation 读取分析。
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class TraceCollector:
    """
    Trace 收集器

    线程安全地将 trace 数据追加写入 jsonl 文件。

    用法::

        collector = TraceCollector("./logs/traces.jsonl")
        trace = TraceContext(operation="retrieval")
        # ... record stages ...
        collector.collect(trace)
    """

    def __init__(self, log_file: Optional[str] = None) -> None:
        """
        初始化 TraceCollector

        Args:
            log_file: jsonl 日志文件路径。为 None 时仅在 DEBUG 级别输出到 stderr，不写文件。
        """
        self._log_file = Path(log_file) if log_file else None
        self._lock = threading.Lock()

        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> Dict[str, Any]:
        """
        收集 trace：调用 finish()，写入文件并返回 trace dict。

        Args:
            trace: 已记录完阶段的 TraceContext 实例

        Returns:
            finish() 产出的可 JSON 序列化 dict
        """
        data = trace.finish()
        self._write(data)
        return data

    def collect_dict(self, data: Dict[str, Any]) -> None:
        """直接写入已序列化的 trace dict（供外部已调用 finish() 的场景使用）"""
        self._write(data)

    def _write(self, data: Dict[str, Any]) -> None:
        line = json.dumps(data, ensure_ascii=False, default=str)

        if self._log_file:
            with self._lock:
                try:
                    with open(self._log_file, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except OSError as e:
                    logger.warning("写入 trace 日志失败 %s: %s", self._log_file, e)

        logger.debug("trace: %s", line)
