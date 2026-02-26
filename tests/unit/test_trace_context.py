"""TraceContext 与 TraceCollector 单元测试"""
import json
import time
from pathlib import Path

import pytest

from src.core.trace.trace_context import StageRecord, TraceContext
from src.core.trace.trace_collector import TraceCollector


# ── TraceContext 基础 ──────────────────────────────────────────────


class TestTraceContextBasic:
    """TraceContext 创建与基础属性"""

    def test_default_trace_id(self) -> None:
        trace = TraceContext(operation="retrieval")
        assert len(trace.trace_id) == 16
        assert trace.operation == "retrieval"

    def test_custom_trace_id(self) -> None:
        trace = TraceContext(operation="ingest", trace_id="abc123")
        assert trace.trace_id == "abc123"

    def test_initial_state_empty(self) -> None:
        trace = TraceContext()
        assert trace.stages == []
        assert trace.metrics == {}


# ── record_stage ──────────────────────────────────────────────────


class TestRecordStage:
    """手动记录阶段"""

    def test_record_with_duration(self) -> None:
        trace = TraceContext(operation="retrieval")
        rec = trace.record_stage("dense_retrieval", duration_ms=42.5, top_k=20)
        assert rec.name == "dense_retrieval"
        assert rec.duration_ms == 42.5
        assert rec.metadata == {"top_k": 20}
        assert len(trace.stages) == 1

    def test_record_with_start_end(self) -> None:
        trace = TraceContext()
        rec = trace.record_stage("fusion", start_ms=10.0, end_ms=25.0)
        assert rec.start_ms == 10.0
        assert rec.end_ms == 25.0
        assert rec.duration_ms == pytest.approx(15.0)

    def test_record_no_time_params(self) -> None:
        trace = TraceContext()
        rec = trace.record_stage("placeholder")
        assert rec.duration_ms == 0.0

    def test_multiple_stages_ordered(self) -> None:
        trace = TraceContext()
        trace.record_stage("a", duration_ms=1.0)
        trace.record_stage("b", duration_ms=2.0)
        trace.record_stage("c", duration_ms=3.0)
        names = [s.name for s in trace.stages]
        assert names == ["a", "b", "c"]


# ── stage() 上下文管理器 ──────────────────────────────────────────


class TestStageContextManager:
    """with trace.stage(...) 自动计时"""

    def test_stage_timer_records_duration(self) -> None:
        trace = TraceContext(operation="test")
        with trace.stage("sleep_stage"):
            time.sleep(0.02)
        assert len(trace.stages) == 1
        rec = trace.stages[0]
        assert rec.name == "sleep_stage"
        assert rec.duration_ms >= 15.0  # ~20ms with tolerance

    def test_stage_timer_with_metadata(self) -> None:
        trace = TraceContext()
        with trace.stage("dense", top_k=20, collection="kb"):
            pass
        rec = trace.stages[0]
        assert rec.metadata["top_k"] == 20
        assert rec.metadata["collection"] == "kb"

    def test_stage_timer_captures_error(self) -> None:
        trace = TraceContext()
        with pytest.raises(ValueError, match="boom"):
            with trace.stage("failing"):
                raise ValueError("boom")
        rec = trace.stages[0]
        assert "error" in rec.metadata
        assert "boom" in rec.metadata["error"]
        assert rec.duration_ms >= 0.0


# ── set_metric / finish ──────────────────────────────────────────


class TestFinish:
    """finish() 输出可 JSON 序列化的 dict"""

    def test_finish_structure(self) -> None:
        trace = TraceContext(operation="retrieval", trace_id="test123")
        trace.record_stage("dense", duration_ms=10.0)
        trace.set_metric("query", "what is RAG")
        trace.set_metric("total_results", 5)

        result = trace.finish()

        assert result["trace_id"] == "test123"
        assert result["operation"] == "retrieval"
        assert isinstance(result["timestamp"], float)
        assert result["total_duration_ms"] >= 0
        assert len(result["stages"]) == 1
        assert result["stages"][0]["name"] == "dense"
        assert result["metrics"]["query"] == "what is RAG"
        assert result["metrics"]["total_results"] == 5

    def test_finish_json_serializable(self) -> None:
        trace = TraceContext(operation="ingest")
        trace.record_stage("load", duration_ms=100.0, source="test.pdf")
        with trace.stage("split"):
            pass
        trace.set_metric("chunk_count", 42)
        result = trace.finish()

        serialized = json.dumps(result, ensure_ascii=False)
        parsed = json.loads(serialized)
        assert parsed["trace_id"] == trace.trace_id
        assert len(parsed["stages"]) == 2

    def test_finish_idempotent(self) -> None:
        trace = TraceContext()
        r1 = trace.finish()
        time.sleep(0.01)
        r2 = trace.finish()
        assert r1["total_duration_ms"] == r2["total_duration_ms"]


# ── StageRecord ──────────────────────────────────────────────────


class TestStageRecord:
    """StageRecord.to_dict()"""

    def test_to_dict_without_metadata(self) -> None:
        rec = StageRecord(name="x", start_ms=0, end_ms=10, duration_ms=10)
        d = rec.to_dict()
        assert d == {"name": "x", "start_ms": 0, "end_ms": 10, "duration_ms": 10}
        assert "metadata" not in d

    def test_to_dict_with_metadata(self) -> None:
        rec = StageRecord(name="y", start_ms=0, end_ms=5, duration_ms=5, metadata={"k": 1})
        d = rec.to_dict()
        assert d["metadata"] == {"k": 1}


# ── TraceCollector ───────────────────────────────────────────────


class TestTraceCollector:
    """TraceCollector 写入 jsonl"""

    def test_collect_writes_jsonl(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        collector = TraceCollector(str(log_file))

        trace = TraceContext(operation="retrieval", trace_id="col_test")
        trace.record_stage("dense", duration_ms=5.0)
        data = collector.collect(trace)

        assert data["trace_id"] == "col_test"
        assert log_file.exists()

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["trace_id"] == "col_test"
        assert parsed["stages"][0]["name"] == "dense"

    def test_collect_appends_multiple(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        collector = TraceCollector(str(log_file))

        for i in range(3):
            trace = TraceContext(trace_id=f"t{i}")
            collector.collect(trace)

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        ids = [json.loads(line)["trace_id"] for line in lines]
        assert ids == ["t0", "t1", "t2"]

    def test_collect_no_file(self) -> None:
        collector = TraceCollector(log_file=None)
        trace = TraceContext(operation="test")
        data = collector.collect(trace)
        assert "trace_id" in data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log_file = tmp_path / "sub" / "dir" / "traces.jsonl"
        collector = TraceCollector(str(log_file))
        trace = TraceContext()
        collector.collect(trace)
        assert log_file.exists()
