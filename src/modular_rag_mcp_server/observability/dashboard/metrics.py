"""Pure Dashboard metric calculations without Streamlit dependencies."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_trace_status(traces: list[dict]) -> dict[str, float | int]:
    """Summarize success from top-level and stage-level trace errors."""
    failed = 0
    for trace in traces:
        has_error = bool(trace.get("error"))
        if not has_error:
            has_error = any(
                bool(stage.get("metadata", {}).get("error"))
                for stage in trace.get("stages", [])
            )
        failed += int(has_error)
    total = len(traces)
    successful = total - failed
    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total * 100 if total else 0.0,
    }


def summarize_overview(traces: list[dict]) -> dict[str, Any]:
    """Calculate overview KPIs from the currently filtered traces."""
    durations = sorted(float(trace.get("total_duration_ms", 0) or 0) for trace in traces)
    stage_totals: defaultdict[str, float] = defaultdict(float)
    for trace in traces:
        for stage in trace.get("stages", []):
            stage_totals[str(stage.get("name", "unknown"))] += float(
                stage.get("duration_ms", 0) or 0
            )
    status = summarize_trace_status(traces)
    return {
        "total": len(traces),
        "p50_ms": _percentile(durations, 0.50),
        "p95_ms": _percentile(durations, 0.95),
        "p99_ms": _percentile(durations, 0.99),
        "error_rate": (status["failed"] / len(traces) * 100) if traces else 0.0,
        "slowest_stage": max(stage_totals.items(), key=lambda item: item[1])[0]
        if stage_totals
        else "-",
    }


def summarize_rag_metrics(trace: dict) -> dict[str, Any]:
    """Extract common RAG latency and generation metrics for the detail view."""
    stage_durations = {
        str(stage.get("name", "")): float(stage.get("duration_ms", 0) or 0)
        for stage in trace.get("stages", [])
    }
    metrics = trace.get("metrics", {})
    return {
        "retrieval_ms": _first_stage_duration(stage_durations, "retrieval", "dense", "sparse"),
        "rerank_ms": _first_stage_duration(stage_durations, "rerank"),
        "final_results": metrics.get("total_results", metrics.get("final_results", "-")),
        "tokens": metrics.get("token_count", metrics.get("total_tokens", "-")),
        "cost_usd": metrics.get("cost_usd", metrics.get("llm_cost_usd", "-")),
    }


def _first_stage_duration(stage_durations: dict[str, float], *terms: str) -> float | str:
    for name, duration in stage_durations.items():
        if any(term in name.lower() for term in terms):
            return duration
    return "-"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return round(values[index], 1)
