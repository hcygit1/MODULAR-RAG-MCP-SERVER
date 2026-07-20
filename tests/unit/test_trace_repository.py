"""Trace repository filtering and bounded loading tests."""

import json
from pathlib import Path

from modular_rag_mcp_server.observability.dashboard.trace_repository import (
    TraceQuery,
    filter_traces,
    load_traces,
)


def test_load_traces_keeps_newest_records_with_invalid_lines_skipped(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    records = [
        {"trace_id": "old", "timestamp": 1},
        {"trace_id": "new", "timestamp": 3},
        {"trace_id": "mid", "timestamp": 2},
    ]
    path.write_text(
        "\n".join([json.dumps(records[0]), "not-json", json.dumps(records[1]), json.dumps(records[2])]),
        encoding="utf-8",
    )

    loaded = load_traces(str(path), limit=2)

    assert [trace["trace_id"] for trace in loaded] == ["new", "mid"]


def test_filter_traces_combines_status_duration_operation_and_text() -> None:
    traces = [
        {
            "trace_id": "slow-error",
            "operation": "retrieval",
            "status": "error",
            "total_duration_ms": 1200,
            "stages": [{"metadata": {"error": "timeout"}}],
        },
        {
            "trace_id": "fast-ok",
            "operation": "retrieval",
            "status": "ok",
            "total_duration_ms": 50,
            "stages": [],
        },
    ]

    result = filter_traces(
        traces,
        TraceQuery(operation="retrieval", status="error", min_duration_ms=1000, text="timeout"),
    )

    assert [trace["trace_id"] for trace in result] == ["slow-error"]
