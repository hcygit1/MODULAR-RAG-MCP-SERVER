"""Dashboard 指标计算测试。"""

from modular_rag_mcp_server.observability.dashboard.metrics import (
    summarize_overview,
    summarize_rag_metrics,
    summarize_trace_status,
)


def test_summarize_trace_status_counts_stage_errors():
    traces = [
        {"stages": [{"metadata": {}}]},
        {"stages": [{"metadata": {"error": "timeout"}}]},
        {"error": "request failed", "stages": []},
    ]

    assert summarize_trace_status(traces) == {
        "total": 3,
        "successful": 1,
        "failed": 2,
        "success_rate": 33.33333333333333,
    }


def test_summarize_trace_status_handles_empty_input():
    assert summarize_trace_status([]) == {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "success_rate": 0.0,
    }


def test_summarize_overview_uses_percentiles_and_slowest_stage():
    traces = [
        {"total_duration_ms": 20, "stages": [{"name": "dense", "duration_ms": 18}]},
        {"total_duration_ms": 100, "stages": [{"name": "rerank", "duration_ms": 80}]},
        {"total_duration_ms": 50, "stages": [{"name": "dense", "duration_ms": 30}]},
    ]

    summary = summarize_overview(traces)

    assert summary["total"] == 3
    assert summary["p50_ms"] == 50.0
    assert summary["p95_ms"] == 100.0
    assert summary["slowest_stage"] == "rerank"


def test_summarize_rag_metrics_reads_stage_and_metric_values():
    trace = {
        "metrics": {"total_results": 6, "token_count": 128, "cost_usd": 0.02},
        "stages": [
            {"name": "dense_retrieval", "duration_ms": 20},
            {"name": "rerank", "duration_ms": 30},
        ],
    }

    assert summarize_rag_metrics(trace) == {
        "retrieval_ms": 20.0,
        "rerank_ms": 30.0,
        "final_results": 6,
        "tokens": 128,
        "cost_usd": 0.02,
    }
