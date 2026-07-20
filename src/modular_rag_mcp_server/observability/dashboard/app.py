"""RAG trace explorer and observability dashboard."""

from __future__ import annotations

import html
import json
from datetime import date, datetime
from pathlib import Path

import altair as alt  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st  # type: ignore[import-not-found]

from modular_rag_mcp_server.core.runtime_paths import get_runtime_paths
from modular_rag_mcp_server.observability.dashboard.metrics import (
    summarize_overview,
    summarize_rag_metrics,
)
from modular_rag_mcp_server.observability.dashboard.trace_repository import (
    TraceQuery,
    filter_traces,
    trace_status,
)
from modular_rag_mcp_server.observability.dashboard.trace_repository import (
    load_traces as load_trace_file,
)

DEFAULT_LOG_FILE = str(get_runtime_paths().log_path)
DEFAULT_TRACE_LIMIT = 1000
STAGE_COLORS = {
    "retrieval": "#2563eb",
    "dense": "#2563eb",
    "sparse": "#0891b2",
    "fusion": "#7c3aed",
    "rerank": "#c026d3",
    "llm": "#d97706",
    "encode": "#059669",
    "store": "#dc2626",
}


def load_traces(log_file: str, limit: int = DEFAULT_TRACE_LIMIT) -> list[dict]:
    """Compatibility wrapper for callers that used the old app helper."""
    return load_trace_file(log_file, limit=limit)


def format_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return "-"
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError, OverflowError):
        return str(timestamp)


def inject_css() -> None:
    """Keep the dashboard dense, neutral and readable for repeated investigation."""
    st.markdown(
        """
        <style>
        :root {
          --surface: #ffffff;
          --surface-muted: #f8fafc;
          --border: #dbe3ec;
          --text: #172033;
          --muted: #5b687a;
          --danger: #b42318;
          --success: #067647;
        }
        .block-container { max-width: 1500px; padding-top: 2rem; }
        .eyebrow { color: var(--muted); font-size: .82rem; letter-spacing: .04em; }
        .trace-header {
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 16px 18px; margin: 8px 0 16px;
        }
        .trace-title { color: var(--text); font-size: 1.15rem; font-weight: 650; }
        .trace-id { color: var(--muted); font-family: monospace; font-size: .82rem; margin-top: 5px; }
        .status-ok { color: var(--success); font-weight: 650; }
        .status-error { color: var(--danger); font-weight: 650; }
        [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); padding: 12px 14px; border-radius: 8px; }
        @media (max-width: 800px) {
          .block-container { padding: 1rem .75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def maybe_auto_refresh(enabled: bool, seconds: int) -> None:
    if not enabled:
        return
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore[import-not-found]
    except ImportError:
        st.caption("自动刷新需要安装 dashboard extra 的刷新依赖。")
        return
    st_autorefresh(interval=seconds * 1000, key="trace-auto-refresh")


def render_overview_kpis(traces: list[dict]) -> None:
    if not traces:
        return
    summary = summarize_overview(traces)
    columns = st.columns(5)
    with columns[0]:
        st.metric("Requests", f"{summary['total']:,}")
    with columns[1]:
        st.metric("Error rate", f"{summary['error_rate']:.1f}%")
    with columns[2]:
        st.metric("P50", f"{summary['p50_ms']:.1f} ms")
    with columns[3]:
        st.metric("P95", f"{summary['p95_ms']:.1f} ms")
    with columns[4]:
        st.metric("Slowest stage", str(summary["slowest_stage"]))


def render_latency_chart(traces: list[dict]) -> None:
    if not traces:
        return
    frame = pd.DataFrame(
        [
            {
                "time": datetime.fromtimestamp(trace["timestamp"]),
                "duration": trace.get("total_duration_ms", 0),
                "operation": trace.get("operation", "unknown"),
                "status": trace_status(trace),
                "trace_id": trace.get("trace_id", ""),
            }
            for trace in traces
            if trace.get("timestamp")
        ]
    )
    if frame.empty:
        st.info("当前 trace 没有可绘制的时间戳。")
        return
    chart = (
        alt.Chart(frame)
        .mark_circle(size=70)
        .encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("duration:Q", title="Latency (ms)", scale=alt.Scale(zero=False)),
            color=alt.Color("status:N", title="Status"),
            tooltip=["trace_id", "operation", "status", "duration"],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_gantt_chart(stages: list[dict]) -> None:
    """Render the trace critical path and stage durations."""
    valid = [stage for stage in stages if "start_ms" in stage and "end_ms" in stage]
    if not valid:
        st.info("该 trace 没有完整的阶段时间数据。")
        return
    min_start = min(stage["start_ms"] for stage in valid)
    rows = []
    for index, stage in enumerate(valid):
        name = str(stage.get("name", f"stage_{index}"))
        rows.append(
            {
                "stage": f"{index + 1:02d}. {name}",
                "name": name,
                "start": stage["start_ms"] - min_start,
                "end": stage["end_ms"] - min_start,
                "duration": stage.get("duration_ms", 0),
            }
        )
    frame = pd.DataFrame(rows)
    chart = (
        alt.Chart(frame)
        .mark_bar(size=22, cornerRadius=3)
        .encode(
            x=alt.X("start:Q", title="Timeline (ms)"),
            x2="end:Q",
            y=alt.Y("stage:N", sort=alt.EncodingSortField(field="start", order="ascending"), title=""),
            color=alt.Color("name:N", title="Stage", legend=None),
            tooltip=["name", "duration", "start", "end"],
        )
        .properties(height=max(220, len(rows) * 38))
    )
    st.altair_chart(chart, use_container_width=True)


def render_stages_table(stages: list[dict]) -> None:
    if not stages:
        st.info("该 trace 无阶段记录。")
        return
    rows = []
    for stage in stages:
        metadata = stage.get("metadata", {})
        rows.append(
            {
                "Stage": stage.get("name", "unknown"),
                "Status": "error" if metadata.get("error") else "ok",
                "Duration (ms)": round(float(stage.get("duration_ms", 0) or 0), 3),
                "Metadata": json.dumps(metadata, ensure_ascii=False),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_rag_metrics(trace: dict) -> None:
    metrics = summarize_rag_metrics(trace)
    columns = st.columns(len(metrics))
    labels = {
        "retrieval_ms": "Retrieval",
        "rerank_ms": "Rerank",
        "final_results": "Final results",
        "tokens": "Tokens",
        "cost_usd": "Cost (USD)",
    }
    for column, (key, value) in zip(columns, metrics.items()):
        with column:
            label = labels.get(key, key)
            display = f"{value:.1f} ms" if isinstance(value, float) and key.endswith("_ms") else str(value)
            st.metric(label, display)


def render_trace_detail(trace: dict) -> None:
    status = trace_status(trace)
    operation = html.escape(str(trace.get("operation", "unknown")))
    trace_id = html.escape(str(trace.get("trace_id", "unknown")))
    status_class = "status-error" if status == "error" else "status-ok"
    st.markdown(
        f"""
        <div class="trace-header">
          <div class="trace-title">{operation} <span class="{status_class}">{status}</span></div>
          <div class="trace-id">Trace ID: {trace_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    with columns[0]:
        st.metric("Duration", f"{float(trace.get('total_duration_ms', 0) or 0):.1f} ms")
    with columns[1]:
        st.metric("Stages", len(trace.get("stages", [])))
    with columns[2]:
        st.metric("Environment", str(trace.get("environment", "-")))
    with columns[3]:
        st.metric("Timestamp", format_timestamp(trace.get("timestamp")))

    st.markdown("#### Waterfall")
    render_gantt_chart(trace.get("stages", []))
    stages = trace.get("stages", [])
    if stages:
        stage_names = [str(stage.get("name", "unknown")) for stage in stages]
        selected_stage = st.selectbox("Inspect stage", options=range(len(stages)), format_func=lambda i: stage_names[i])
        st.json(stages[selected_stage].get("metadata", {}))

    tab_stages, tab_rag, tab_raw = st.tabs(["Stages", "RAG metrics", "Raw JSON"])
    with tab_stages:
        render_stages_table(stages)
    with tab_rag:
        render_rag_metrics(trace)
    with tab_raw:
        st.download_button(
            "Download trace JSON",
            data=json.dumps(trace, ensure_ascii=False, indent=2),
            file_name=f"trace-{trace_id}.json",
            mime="application/json",
        )
        st.json(trace)


def _trace_option(trace: dict) -> str:
    status = trace_status(trace).upper()
    operation = str(trace.get("operation", "unknown"))
    duration = float(trace.get("total_duration_ms", 0) or 0)
    trace_id = str(trace.get("trace_id", ""))[:12]
    return f"{status:<5}  {operation:<16}  {duration:>8.1f} ms  {trace_id}"


def main() -> None:
    st.set_page_config(page_title="RAG Observability", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    st.title("RAG Observability")
    st.caption("Trace Explorer / Retrieval, rerank and generation diagnostics")

    with st.sidebar:
        st.subheader("Data source")
        log_file = st.text_input("Trace file", value=DEFAULT_LOG_FILE)
        trace_limit = int(st.number_input("Recent traces", min_value=100, max_value=10000, value=DEFAULT_TRACE_LIMIT, step=100))
        auto_refresh = st.checkbox("Auto refresh", value=False)
        refresh_seconds = int(st.selectbox("Refresh interval", options=[10, 30, 60], index=1))
        st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")
        maybe_auto_refresh(auto_refresh, refresh_seconds)

    all_traces = load_traces(log_file, limit=trace_limit)
    if not all_traces:
        st.info(f"No trace data found in {Path(log_file)}.")
        return

    st.subheader("Trace Explorer")
    operations = sorted({str(trace.get("operation", "unknown")) for trace in all_traces})
    dates = [datetime.fromtimestamp(float(trace["timestamp"])).date() for trace in all_traces if trace.get("timestamp")]
    query_columns = st.columns([2.2, 1.1, 1.1, 1.0])
    with query_columns[0]:
        text_query = st.text_input("Search trace, error or attribute", placeholder="trace id, timeout, collection...")
    with query_columns[1]:
        operation = st.selectbox("Operation", options=["All", *operations])
    with query_columns[2]:
        status = st.selectbox("Status", options=["All", "ok", "error"])
    with query_columns[3]:
        min_duration = st.number_input("Min ms", min_value=0.0, value=0.0, step=100.0)

    date_columns = st.columns([1, 1, 2])
    with date_columns[0]:
        start_date = st.date_input("From", value=min(dates) if dates else date.today())
    with date_columns[1]:
        end_date = st.date_input("To", value=max(dates) if dates else date.today())
    with date_columns[2]:
        st.caption("Filters apply to the bounded trace set loaded from the JSONL file.")

    query = TraceQuery(
        operation=None if operation == "All" else operation,
        status=None if status == "All" else status,
        text=text_query or None,
        min_duration_ms=min_duration or None,
        start_date=start_date,
        end_date=end_date,
    )
    traces = filter_traces(all_traces, query)
    st.caption(f"Showing {len(traces):,} of {len(all_traces):,} loaded traces")
    if not traces:
        st.warning("No traces match the current filters.")
        return

    render_overview_kpis(traces)
    with st.expander("Latency trend", expanded=False):
        render_latency_chart(traces)

    list_column, detail_column = st.columns([0.95, 1.8], gap="large")
    with list_column:
        st.markdown("#### Trace list")
        selected_index = st.selectbox(
            "Select trace",
            options=range(len(traces)),
            format_func=lambda index: _trace_option(traces[index]),
            label_visibility="collapsed",
        )
        overview = pd.DataFrame(
            [
                {
                    "Status": trace_status(trace),
                    "Operation": trace.get("operation", "unknown"),
                    "Duration": f"{float(trace.get('total_duration_ms', 0) or 0):.1f} ms",
                    "Trace": str(trace.get("trace_id", ""))[:12],
                }
                for trace in traces
            ]
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)
    with detail_column:
        st.markdown("#### Trace details")
        render_trace_detail(traces[selected_index])


if __name__ == "__main__":
    main()
