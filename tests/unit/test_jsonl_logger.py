"""结构化日志 logger（JSON Lines）单元测试"""
import json
from pathlib import Path

import pytest

from src.core.trace.trace_context import TraceContext
from src.observability.logger import (
    get_logger,
    get_trace_collector,
    init_trace_collector,
    reset_trace_collector,
)


@pytest.fixture(autouse=True)
def _reset_collector():
    """每个测试前后重置全局 collector，避免测试间相互影响"""
    reset_trace_collector()
    yield
    reset_trace_collector()


# ── get_logger 基础 ──────────────────────────────────────────────


class TestGetLogger:
    """get_logger 仍正常工作"""

    def test_returns_logger(self) -> None:
        logger = get_logger("test_module")
        assert logger.name == "test_module"

    def test_outputs_to_stderr(self) -> None:
        import sys
        logger = get_logger("stderr_check")
        assert any(
            hasattr(h, "stream") and h.stream is sys.stderr
            for h in logger.handlers
        )


# ── init / get trace collector ───────────────────────────────────


class TestTraceCollectorLifecycle:
    """init_trace_collector / get_trace_collector 单例管理"""

    def test_init_creates_collector(self, tmp_path: Path) -> None:
        log_file = str(tmp_path / "traces.jsonl")
        collector = init_trace_collector(log_file)
        assert collector is not None
        assert collector._log_file == Path(log_file)

    def test_init_returns_same_instance(self, tmp_path: Path) -> None:
        log_file = str(tmp_path / "traces.jsonl")
        c1 = init_trace_collector(log_file)
        c2 = init_trace_collector(log_file)
        assert c1 is c2

    def test_init_replaces_on_different_path(self, tmp_path: Path) -> None:
        c1 = init_trace_collector(str(tmp_path / "a.jsonl"))
        c2 = init_trace_collector(str(tmp_path / "b.jsonl"))
        assert c1 is not c2

    def test_get_returns_default_if_not_init(self) -> None:
        collector = get_trace_collector()
        assert collector is not None
        assert collector._log_file is None

    def test_get_returns_init_instance(self, tmp_path: Path) -> None:
        log_file = str(tmp_path / "traces.jsonl")
        expected = init_trace_collector(log_file)
        assert get_trace_collector() is expected

    def test_reset_clears_collector(self, tmp_path: Path) -> None:
        init_trace_collector(str(tmp_path / "traces.jsonl"))
        reset_trace_collector()
        collector = get_trace_collector()
        assert collector._log_file is None


# ── 核心验收：写入 trace 后 jsonl 文件新增一行合法 JSON ──────────


class TestJsonlWrite:
    """验收标准：写入一条 trace 后文件新增一行合法 JSON"""

    def test_single_trace_writes_one_line(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        collector = init_trace_collector(str(log_file))

        trace = TraceContext(operation="retrieval", trace_id="jsonl_test")
        trace.record_stage("dense", duration_ms=10.0, top_k=20)
        trace.set_metric("query", "what is RAG")
        collector.collect(trace)

        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        assert parsed["trace_id"] == "jsonl_test"
        assert parsed["operation"] == "retrieval"
        assert parsed["stages"][0]["name"] == "dense"
        assert parsed["metrics"]["query"] == "what is RAG"

    def test_multiple_traces_append(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        collector = init_trace_collector(str(log_file))

        for i in range(3):
            trace = TraceContext(operation="ingest", trace_id=f"multi_{i}")
            trace.record_stage("load", duration_ms=float(i))
            collector.collect(trace)

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["trace_id"] == f"multi_{i}"

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        collector = init_trace_collector(str(log_file))

        trace = TraceContext(operation="test")
        with trace.stage("step_a"):
            pass
        with trace.stage("step_b", key="val"):
            pass
        trace.set_metric("count", 42)
        collector.collect(trace)

        content = log_file.read_text(encoding="utf-8").strip()
        for line in content.split("\n"):
            data = json.loads(line)
            assert "trace_id" in data
            assert "stages" in data
            assert isinstance(data["stages"], list)

    def test_get_collector_then_write(self, tmp_path: Path) -> None:
        """通过 get_trace_collector 获取实例后写入"""
        log_file = str(tmp_path / "traces.jsonl")
        init_trace_collector(log_file)

        collector = get_trace_collector()
        trace = TraceContext(trace_id="via_get")
        collector.collect(trace)

        parsed = json.loads(Path(log_file).read_text(encoding="utf-8").strip())
        assert parsed["trace_id"] == "via_get"

    def test_no_file_mode_does_not_crash(self) -> None:
        """log_file=None 时仅 debug 输出，不崩溃"""
        collector = init_trace_collector(log_file=None)
        trace = TraceContext(trace_id="no_file")
        data = collector.collect(trace)
        assert data["trace_id"] == "no_file"
