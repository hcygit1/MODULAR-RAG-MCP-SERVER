"""Bounded trace loading and explorer filters for the Dashboard."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceQuery:
    """Composable filters used by the trace explorer."""

    operation: str | None = None
    status: str | None = None
    text: str | None = None
    min_duration_ms: float | None = None
    max_duration_ms: float | None = None
    start_date: date | None = None
    end_date: date | None = None


def load_traces(log_file: str, limit: int = 1000) -> list[dict[str, Any]]:
    """Load at most the newest ``limit`` valid traces from a JSONL file."""
    if limit <= 0:
        return []
    path = Path(log_file)
    if not path.exists():
        return []

    records: deque[dict[str, Any]] = deque(maxlen=limit)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return sorted(records, key=lambda trace: trace.get("timestamp", 0), reverse=True)


def trace_status(trace: dict[str, Any]) -> str:
    """Return ``ok`` or ``error`` while supporting legacy trace records."""
    explicit = str(trace.get("status", "")).lower()
    if explicit in {"ok", "error"}:
        return explicit
    if trace.get("error"):
        return "error"
    for stage in trace.get("stages", []):
        if stage.get("metadata", {}).get("error"):
            return "error"
    return "ok"


def filter_traces(traces: list[dict[str, Any]], query: TraceQuery) -> list[dict[str, Any]]:
    """Apply explorer filters without mutating the source list."""
    text = query.text.strip().lower() if query.text else ""
    result: list[dict[str, Any]] = []
    for trace in traces:
        timestamp = trace.get("timestamp")
        trace_date = _trace_date(timestamp)
        duration = float(trace.get("total_duration_ms", 0) or 0)
        searchable = " ".join(
            [
                str(trace.get("trace_id", "")),
                str(trace.get("operation", "")),
                str(trace.get("error", "")),
                json.dumps(trace.get("attributes", {}), ensure_ascii=False),
                json.dumps(trace.get("stages", []), ensure_ascii=False),
            ]
        ).lower()
        if query.operation and trace.get("operation") != query.operation:
            continue
        if query.status and trace_status(trace) != query.status:
            continue
        if text and text not in searchable:
            continue
        if query.min_duration_ms is not None and duration < query.min_duration_ms:
            continue
        if query.max_duration_ms is not None and duration > query.max_duration_ms:
            continue
        if query.start_date and (trace_date is None or trace_date < query.start_date):
            continue
        if query.end_date and (trace_date is None or trace_date > query.end_date):
            continue
        result.append(trace)
    return result


def _trace_date(timestamp: Any) -> date | None:
    try:
        return datetime.fromtimestamp(float(timestamp)).date()
    except (TypeError, ValueError, OSError, OverflowError):
        return None
