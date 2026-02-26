"""
日志与 Trace 模块

提供两层能力：
1. get_logger() — 统一的 Python logging 接口，输出到 stderr（避免污染 stdout 的 MCP 消息）。
2. get_trace_collector() / init_trace_collector() — 全局 TraceCollector 单例，
   将 TraceContext 写入 JSON Lines 文件，供 Dashboard / Evaluation 使用。
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

from src.core.trace.trace_collector import TraceCollector

# ── Python logging ────────────────────────────────────────────────


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取 logger 实例

    Args:
        name: logger 名称，默认为 None（使用调用模块名）

    Returns:
        logging.Logger: 配置好的 logger 实例，输出到 stderr
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# ── Trace Collector 单例 ──────────────────────────────────────────

_collector_lock = threading.Lock()
_collector: Optional[TraceCollector] = None


def init_trace_collector(log_file: Optional[str] = None) -> TraceCollector:
    """
    初始化全局 TraceCollector 单例。

    首次调用时创建实例；后续调用如果 log_file 相同则返回已有实例，
    如果 log_file 不同则替换为新实例。

    Args:
        log_file: jsonl 日志文件路径（如 "./logs/traces.jsonl"）。
                  为 None 时仅在 DEBUG 级别输出到 stderr，不写文件。

    Returns:
        全局 TraceCollector 实例
    """
    global _collector
    with _collector_lock:
        if _collector is None or _should_replace(log_file):
            _collector = TraceCollector(log_file=log_file)
        return _collector


def get_trace_collector() -> TraceCollector:
    """
    获取全局 TraceCollector 单例。

    如果尚未调用 init_trace_collector()，则自动创建一个不写文件的默认实例。

    Returns:
        全局 TraceCollector 实例
    """
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = TraceCollector(log_file=None)
    return _collector


def _should_replace(log_file: Optional[str]) -> bool:
    """判断是否需要替换现有 collector（log_file 路径变化时替换）"""
    if _collector is None:
        return True
    existing = str(_collector._log_file) if _collector._log_file else None
    return existing != log_file


def reset_trace_collector() -> None:
    """重置全局 TraceCollector（供测试使用）"""
    global _collector
    with _collector_lock:
        _collector = None
