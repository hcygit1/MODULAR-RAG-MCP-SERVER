"""Pure Dashboard metric calculations without Streamlit dependencies."""

from __future__ import annotations


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
