"""Dashboard 指标计算测试。"""

from modular_rag_mcp_server.observability.dashboard.metrics import summarize_trace_status


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
