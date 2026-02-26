"""
Trace 模块

提供请求级链路追踪能力：TraceContext 记录各阶段耗时与指标，
TraceCollector 负责收集并持久化 trace 数据。
"""

from src.core.trace.trace_context import StageRecord, TraceContext

__all__ = ["TraceContext", "StageRecord"]
