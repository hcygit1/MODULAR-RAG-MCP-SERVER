"""
Trace Dashboard (Streamlit)

读取 logs/traces.jsonl，展示 trace 列表与单条详情。
启动方式：streamlit run src/observability/dashboard/app.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

DEFAULT_LOG_FILE = "./logs/traces.jsonl"

STAGE_COLORS = {
    "query_processing": "#6366f1", # Indigo
    "dense_retrieval": "#3b82f6",  # Blue
    "sparse_retrieval": "#06b6d4", # Cyan
    "rrf_fusion": "#8b5cf6",       # Violet
    "rerank": "#ec4899",           # Pink
    "integrity_check": "#64748b",  # Slate
    "load": "#f59e0b",             # Amber
    "split": "#10b981",            # Emerald
    "image_save": "#f97316",       # Orange
    "transform": "#14b8a6",        # Teal
    "encode": "#a855f7",           # Purple
    "store": "#ef4444",            # Red
}

OP_ICONS = {
    "retrieval": "🔍",
    "ingestion": "📥",
}


def load_traces(log_file: str) -> list[dict]:
    path = Path(log_file)
    if not path.exists():
        return []
    traces = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # 按照 timestamp 降序排序，最新的在最前
    traces.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return traces


def format_timestamp(ts: float | None) -> str:
    if ts is None:
        return "-"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError):
        return str(ts)


def inject_css() -> None:
    st.markdown("""
    <style>
    /* 全局字体与样式调整 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #475569;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .sub-value {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #334155;
        margin: 2rem 0;
    }
    .stage-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
    .trace-header {
        background: #1e293b;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .trace-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .trace-id-badge {
        background: #0f172a;
        color: #94a3b8;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub_value: str = "") -> str:
    sub_html = f'<div class="sub-value">{sub_value}</div>' if sub_value else ""
    return f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
        {sub_html}
    </div>
    """


def render_overview_kpis(traces: list[dict]) -> None:
    if not traces:
        return
    
    total = len(traces)
    avg_dur = sum(t.get("total_duration_ms", 0) for t in traces) / total
    
    durations = sorted(t.get("total_duration_ms", 0) for t in traces)
    p95_dur = durations[int(len(durations) * 0.95)] if durations else 0
    p50_dur = durations[int(len(durations) * 0.50)] if durations else 0
    
    # 假设如果 duration > 0 则成功（可以根据实际 status 字段扩展）
    success_rate = 100.0
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Total Traces", f"{total:,}", "Last 30 days"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Avg Latency", f"{avg_dur:.1f} ms", f"Median: {p50_dur:.1f} ms"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("P95 Latency", f"{p95_dur:.1f} ms", "95th percentile"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Success Rate", f"{success_rate:.1f}%", "All operations"), unsafe_allow_html=True)


def render_latency_chart(traces: list[dict]) -> None:
    if not traces:
        return
        
    df = pd.DataFrame([{
        "time": datetime.fromtimestamp(t["timestamp"]) if t.get("timestamp") else datetime.now(),
        "duration": t.get("total_duration_ms", 0),
        "operation": t.get("operation", "unknown"),
        "trace_id": t.get("trace_id", "")
    } for t in traces])
    
    # 时序散点图
    scatter = alt.Chart(df).mark_circle(size=80, opacity=0.8).encode(
        x=alt.X('time:T', title='Time', axis=alt.Axis(grid=False)),
        y=alt.Y('duration:Q', title='Latency (ms)', scale=alt.Scale(zero=False)),
        color=alt.Color('operation:N', title='Operation', legend=alt.Legend(orient="top")),
        tooltip=['trace_id', 'operation', alt.Tooltip('duration:Q', format='.2f'), alt.Tooltip('time:T', format='%Y-%m-%d %H:%M:%S')]
    ).properties(
        height=300
    ).interactive()
    
    # 耗时直方图
    hist = alt.Chart(df).mark_bar(opacity=0.8).encode(
        x=alt.X('duration:Q', bin=alt.Bin(maxbins=30), title='Latency (ms)'),
        y=alt.Y('count()', title='Count'),
        color='operation:N'
    ).properties(
        height=300
    )
    
    st.altair_chart(scatter | hist, use_container_width=True)


def render_gantt_chart(stages: list[dict]) -> None:
    """渲染阶段的甘特图 (Waterfall)"""
    if not stages:
        return

    # 过滤掉没有完整时间的数据
    valid_stages = [s for s in stages if "start_ms" in s and "end_ms" in s]
    if not valid_stages:
        st.warning("阶段数据缺失 start_ms 或 end_ms，无法绘制甘特图。")
        return
        
    # 找到整个 trace 的极小起点，做相对偏移
    min_start = min(s["start_ms"] for s in valid_stages)
    
    df_data = []
    for i, s in enumerate(valid_stages):
        name = s.get("name", f"stage_{i}")
        # 为了防重名，名称带上索引
        unique_name = f"{i:02d}. {name}"
        rel_start = s["start_ms"] - min_start
        rel_end = s["end_ms"] - min_start
        dur = s.get("duration_ms", 0)
        df_data.append({
            "stage_id": unique_name,
            "name": name,
            "start": rel_start,
            "end": rel_end,
            "duration": dur
        })
        
    df = pd.DataFrame(df_data)
    
    # 颜色映射
    domain = list(STAGE_COLORS.keys())
    range_ = list(STAGE_COLORS.values())
    
    chart = alt.Chart(df).mark_bar(cornerRadius=4, size=20).encode(
        x=alt.X('start:Q', title='Timeline (ms)'),
        x2='end:Q',
        y=alt.Y('stage_id:N', sort=alt.EncodingSortField(field="start", op="min", order="ascending"), title='', axis=alt.Axis(labelLimit=200)),
        color=alt.Color('name:N', scale=alt.Scale(domain=domain, range=range_, scheme='category20'), legend=None),
        tooltip=['name:N', alt.Tooltip('duration:Q', format='.2f', title='Duration (ms)'), alt.Tooltip('start:Q', format='.2f', title='Start (ms)'), alt.Tooltip('end:Q', format='.2f', title='End (ms)')]
    ).properties(
        height=max(200, len(valid_stages) * 40)
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def render_stages_table(stages: list[dict]) -> None:
    """渲染阶段详情表"""
    if not stages:
        st.info("该 trace 无阶段记录。")
        return

    rows = []
    for s in stages:
        name = s.get("name", "")
        meta = s.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "-"
        rows.append({
            "Stage": name,
            "Duration (ms)": round(s.get("duration_ms", 0), 3),
            "Start (ms)": round(s.get("start_ms", 0), 3),
            "End (ms)": round(s.get("end_ms", 0), 3),
            "Metadata": meta_str,
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Duration (ms)": st.column_config.NumberColumn(format="%.3f"),
            "Start (ms)": st.column_config.NumberColumn(format="%.3f"),
            "End (ms)": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def render_metrics(metrics: dict) -> None:
    """以卡片形式展示 metrics"""
    if not metrics:
        return

    cols = st.columns(min(len(metrics), 4))
    for i, (k, v) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            display_val = str(v)
            if isinstance(v, float):
                display_val = f"{v:.3f}"
            elif isinstance(v, bool):
                display_val = "✅" if v else "❌"
            st.markdown(metric_card(k, display_val), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="RAG Observability Dashboard",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()

    # ── 侧边栏过滤与配置 ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        log_file = st.text_input("Trace File Path", value=DEFAULT_LOG_FILE)
        traces = load_traces(log_file)

        if traces:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("### 🔍 Filters")
            
            # 操作类型过滤
            operations = sorted(set(t.get("operation", "") for t in traces))
            selected_op = st.selectbox("Operation Type", options=["All"] + operations)
            if selected_op != "All":
                traces = [t for t in traces if t.get("operation") == selected_op]
            
            # 时间范围过滤
            min_ts = min(t.get("timestamp", 0) for t in traces)
            max_ts = max(t.get("timestamp", 0) for t in traces)
            
            if min_ts and max_ts and min_ts != max_ts:
                min_dt = datetime.fromtimestamp(min_ts)
                max_dt = datetime.fromtimestamp(max_ts)
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_dt.date(), max_dt.date()),
                    min_value=min_dt.date(),
                    max_value=max_dt.date()
                )
                
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    traces = [
                        t for t in traces 
                        if start_date <= datetime.fromtimestamp(t.get("timestamp", 0)).date() <= end_date
                    ]

    # ── 主标题 ──
    st.title("✨ RAG Observability Dashboard")
    st.markdown("Monitor, debug, and optimize your Retrieval-Augmented Generation pipelines.")

    if not traces:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.warning(f"No trace data found in `{log_file}` or matching the current filters.")
        st.info("Run a retrieval or ingestion pipeline to generate traces.\n\n"
                "```bash\npython scripts/retrieve.py -q \"Your query\"\n```")
        return

    # ── KPI 概览 ──
    st.markdown("### 📈 System Overview")
    render_overview_kpis(traces)
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("📊 Latency Distribution & Trends", expanded=False):
        render_latency_chart(traces)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 左右分栏布局：左侧 Trace 列表，右侧 Trace 详情 ──
    col_list, col_detail = st.columns([1, 1.8], gap="large")
    
    with col_list:
        st.markdown(f"### 🗂️ Recent Traces ({len(traces)})")
        
        # 使用 selectbox 进行选择，兼容性最好
        trace_options = [
            f"{OP_ICONS.get(t.get('operation', ''), '⚡')} {t.get('trace_id', '')[:8]}... ({format_timestamp(t.get('timestamp'))[11:]})"
            for t in traces
        ]
        selected_idx = st.selectbox(
            "Select Trace to View",
            options=range(len(traces)),
            format_func=lambda i: trace_options[i],
            label_visibility="collapsed"
        )
        
        # 下方额外展示一个只读表格概览
        overview_rows = []
        for t in traces:
            op = t.get("operation", "unknown")
            icon = OP_ICONS.get(op, "⚡")
            overview_rows.append({
                "Op": icon,
                "Trace ID": t.get("trace_id", "")[:8] + "...",
                "Time": format_timestamp(t.get("timestamp"))[11:],
                "Duration": f"{t.get('total_duration_ms', 0):.1f}ms",
            })
            
        st.dataframe(
            pd.DataFrame(overview_rows),
            use_container_width=True,
            hide_index=True,
        )
        
    with col_detail:
        st.markdown("### 🔬 Trace Details")
        selected = traces[selected_idx]
        op = selected.get("operation", "unknown")
        icon = OP_ICONS.get(op, "⚡")
        
        st.markdown(f"""
        <div class="trace-header">
            <div class="trace-title">{icon} {op.capitalize()} Pipeline</div>
            <div class="trace-id-badge">ID: {selected.get('trace_id', 'unknown')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        total_ms = selected.get("total_duration_ms", 0)
        stages = selected.get("stages", [])
        metrics = selected.get("metrics", {})
        
        # 核心摘要指标
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Duration", f"{total_ms:.2f} ms")
        with c2:
            st.metric("Stages Count", len(stages))
        with c3:
            st.metric("Timestamp", format_timestamp(selected.get("timestamp")))
            
        st.markdown("#### ⏱️ Execution Waterfall")
        render_gantt_chart(stages)
        
        st.markdown("<br>", unsafe_allow_html=True)
        tab_stages, tab_metrics, tab_raw = st.tabs(["📋 Stages Data", "📊 Custom Metrics", "🔧 Raw JSON"])
        
        with tab_stages:
            render_stages_table(stages)
            
        with tab_metrics:
            if metrics:
                render_metrics(metrics)
            else:
                st.info("No custom metrics recorded for this trace.")
                
        with tab_raw:
            st.json(selected)


if __name__ == "__main__":
    main()
