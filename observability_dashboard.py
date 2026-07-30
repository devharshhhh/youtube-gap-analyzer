import streamlit as st
import json
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Pipeline Observability", layout="wide")
st.title("📊 Pipeline Observability Dashboard")
st.caption("Internal tooling — latency, cost, and quality tracking for the Content Research Agent")

LOG_FILE = "observability/query_log.jsonl"
BASELINE_FILE = "eval/baseline_scores.json"


# ---- Load query log ----
def load_query_log():
    entries = []
    if not Path(LOG_FILE).exists():
        return pd.DataFrame()
    with open(LOG_FILE) as f:
        for line in f:
            entries.append(json.loads(line))
    return pd.DataFrame(entries)


df = load_query_log()

if df.empty:
    st.warning("No query log data found yet. Run some pipeline queries first (e.g. `python bulk_latency_test.py`).")
else:
    st.markdown(f"### Logged Queries: {len(df)}")

    # ---- Latency section ----
    st.markdown("## ⏱️ Latency")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("p50", f"{df['total_seconds'].quantile(0.5):.3f}s")
    col2.metric("p95", f"{df['total_seconds'].quantile(0.95):.3f}s")
    col3.metric("p99", f"{df['total_seconds'].quantile(0.99):.3f}s")
    col4.metric("avg", f"{df['total_seconds'].mean():.3f}s")

    st.line_chart(df["total_seconds"], height=250)

    with st.expander("Per-stage timing breakdown"):
        stage_data = pd.json_normalize(df["timings_seconds"])
        if not stage_data.empty:
            st.bar_chart(stage_data.mean())

    st.divider()

    # ---- Query log table ----
    st.markdown("## 🔍 Raw Query Log")
    display_cols = ["timestamp", "topic", "query", "total_seconds"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols].sort_values("timestamp", ascending=False), use_container_width=True)


# ---- Retrieval quality section ----
st.divider()
st.markdown("## 🎯 Retrieval Quality (Eval Gate Baseline)")

if Path(BASELINE_FILE).exists():
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    quality_df = pd.DataFrame(baseline).T
    quality_df.columns = ["Recall@5", "MRR"]
    quality_df["Recall@5"] = (quality_df["Recall@5"] * 100).round(1).astype(str) + "%"
    quality_df["MRR"] = quality_df["MRR"].round(4)

    st.dataframe(quality_df, use_container_width=True)

    st.caption("Run `python -m eval.gate_check` to verify current performance against this baseline.")
else:
    st.info("No baseline found. Run `python -m eval.save_baseline` first.")


# ---- Cost section ----
# ---- Cost section ----
st.divider()
st.markdown("## 💰 Cost Tracking")

if not df.empty and "token_counts" in df.columns:
    cost_df = pd.json_normalize(df["token_counts"]).dropna(subset=["total_cost_usd"])

    if cost_df.empty:
        st.info("No cost data logged yet. Run `python graph_pipeline.py` to log a query with cost tracking.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total cost logged", f"${cost_df['total_cost_usd'].sum():.6f}")
        col2.metric("Avg cost per query", f"${cost_df['total_cost_usd'].mean():.6f}")
        col3.metric("Queries with cost data", len(cost_df))

        st.bar_chart(cost_df["total_cost_usd"], height=250)

        with st.expander("Token usage breakdown"):
            token_cols = ["input_tokens", "output_tokens"]
            available_token_cols = [c for c in token_cols if c in cost_df.columns]
            if available_token_cols:
                st.bar_chart(cost_df[available_token_cols])
else:
    st.info("No cost data logged yet. Run `python graph_pipeline.py` to log a query with cost tracking.")