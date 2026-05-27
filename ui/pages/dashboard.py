# ui/pages/dashboard.py
# Dashboard tab: sentiment comparison, topic breakdown, risk signals, review volume.

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.state import AnalysisOutput

# ── Colour palette ─────────────────────────────────────────────────────────────
_POS_DARK   = "#27ae60"
_POS_LIGHT  = "#a9dfbf"
_NEU_DARK   = "#7f8c8d"
_NEU_LIGHT  = "#d5d8dc"
_NEG_DARK   = "#c0392b"
_NEG_LIGHT  = "#f1948a"

_RISK_COLOUR = {
    "LOW":      "#27ae60",
    "MEDIUM":   "#f39c12",
    "HIGH":     "#e67e22",
    "CRITICAL": "#c0392b",
}
_RISK_VALUE = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def render_dashboard(result: dict) -> None:
    """Render the full dashboard from a completed pipeline result."""
    analysis: AnalysisOutput | None = result.get("analysis")

    if analysis is None:
        st.info("No analysis available yet. Run the pipeline from the **Updates** tab.")
        return

    pre  = analysis.pre_sentiment
    post = analysis.post_sentiment

    # ── Key metrics row ───────────────────────────────────────────────────────
    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    delta_pct = f"{analysis.sentiment_delta:+.3f}"
    delta_col = "normal" if analysis.sentiment_delta >= 0 else "inverse"

    c1.metric(
        "Post-update Reviews",
        f"{post.n_reviews:,}",
        delta=f"{post.n_reviews - pre.n_reviews:+,} vs baseline",
    )
    c2.metric(
        "Sentiment Delta",
        delta_pct,
        delta=delta_pct,
        delta_color=delta_col,
    )
    c3.metric(
        "Negative %",
        f"{post.negative_pct:.1%}",
        delta=f"{(post.negative_pct - pre.negative_pct):+.1%} vs baseline",
        delta_color="inverse",
    )
    c4.metric(
        "Overall Risk",
        analysis.overall_risk,
    )

    st.divider()

    # ── Row 1: Sentiment comparison + Risk gauge ──────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Sentiment Distribution")
        fig = go.Figure(data=[
            go.Bar(
                name="Pre-Update",
                x=["Positive", "Neutral", "Negative"],
                y=[pre.positive_pct, pre.neutral_pct, pre.negative_pct],
                marker_color=[_POS_LIGHT, _NEU_LIGHT, _NEG_LIGHT],
                text=[f"{v:.1%}" for v in [pre.positive_pct, pre.neutral_pct, pre.negative_pct]],
                textposition="inside",
            ),
            go.Bar(
                name="Post-Update",
                x=["Positive", "Neutral", "Negative"],
                y=[post.positive_pct, post.neutral_pct, post.negative_pct],
                marker_color=[_POS_DARK, _NEU_DARK, _NEG_DARK],
                text=[f"{v:.1%}" for v in [post.positive_pct, post.neutral_pct, post.negative_pct]],
                textposition="inside",
            ),
        ])
        fig.update_layout(
            barmode="group",
            height=320,
            margin=dict(t=20, b=20, l=0, r=0),
            legend=dict(orientation="h", y=-0.15),
            yaxis=dict(tickformat=".0%", range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk Level")
        risk_val   = _RISK_VALUE[analysis.overall_risk]
        risk_color = _RISK_COLOUR[analysis.overall_risk]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_val,
            number={"suffix": f"  {analysis.overall_risk}", "font": {"size": 20}},
            gauge={
                "axis": {
                    "range": [0, 4],
                    "tickvals": [1, 2, 3, 4],
                    "ticktext": ["LOW", "MED", "HIGH", "CRIT"],
                },
                "bar": {"color": risk_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 1.5], "color": "#d5f5e3"},
                    {"range": [1.5, 2.5], "color": "#fef9e7"},
                    {"range": [2.5, 3.5], "color": "#fdebd0"},
                    {"range": [3.5, 4],   "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": risk_color, "width": 4},
                    "thickness": 0.75,
                    "value": risk_val,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig.update_layout(height=320, margin=dict(t=20, b=20, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 2: Topic breakdown + Risk signals ─────────────────────────────────
    left2, right2 = st.columns([3, 2])

    with left2:
        st.subheader("Topic Breakdown (Post-Update)")
        if analysis.top_topics:
            topics_df = pd.DataFrame([t.model_dump() for t in analysis.top_topics])
            fig = go.Figure(go.Bar(
                x=topics_df["pct"],
                y=topics_df["label"],
                orientation="h",
                text=[f"{p:.1%}  ({c})" for p, c in zip(topics_df["pct"], topics_df["count"])],
                textposition="inside",
                marker_color="#3498db",
            ))
            fig.update_layout(
                height=max(250, len(analysis.top_topics) * 38),
                margin=dict(t=10, b=10, l=0, r=0),
                xaxis=dict(tickformat=".0%"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Topic data unavailable.")

    with right2:
        st.subheader("Risk Signals")
        if analysis.risk_signals:
            for sig in analysis.risk_signals:
                severity_emoji = {
                    "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
                }.get(sig.severity, "⚪")
                with st.expander(f"{severity_emoji} [{sig.rule}] {sig.severity}"):
                    st.write(sig.description)
                    st.caption(f"Value: **{sig.value}** · Threshold: {sig.threshold}")
        else:
            st.success("✅ No risk signals triggered.")

    # ── Sentiment over time (if raw reviews available) ────────────────────────
    pre_df  = result.get("pre_reviews")
    post_df = result.get("cleaned_reviews")

    if pre_df is not None and post_df is not None and "vader_compound" in post_df.columns:
        st.divider()
        st.subheader("Sentiment Over Time")

        update_date = result.get("update_date") if isinstance(result.get("update_date"), pd.Timestamp) \
                      else pd.Timestamp(result["update_date"])

        combined_parts = []
        if not pre_df.empty and "vader_compound" in pre_df.columns:
            tmp = pre_df[["timestamp", "vader_compound"]].copy()
            tmp["window"] = "Pre-Update"
            combined_parts.append(tmp)
        if not post_df.empty:
            tmp = post_df[["timestamp", "vader_compound"]].copy()
            tmp["window"] = "Post-Update"
            combined_parts.append(tmp)

        if combined_parts:
            combined = pd.concat(combined_parts, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["timestamp"]).dt.date
            daily = (
                combined.groupby(["date", "window"])["vader_compound"]
                .mean()
                .reset_index()
            )

            fig = go.Figure()
            for window, color in [("Pre-Update", "#95a5a6"), ("Post-Update", "#3498db")]:
                subset = daily[daily["window"] == window]
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset["date"], y=subset["vader_compound"],
                        mode="lines+markers", name=window,
                        line=dict(color=color, width=2),
                    ))

            # Update date marker
            fig.add_vline(
                x=str(update_date.date()),
                line_dash="dash", line_color="#e74c3c",
                annotation_text="Update", annotation_position="top right",
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#bdc3c7", opacity=0.5)
            fig.update_layout(
                height=300,
                margin=dict(t=20, b=20, l=0, r=0),
                yaxis=dict(title="Compound Score", range=[-1, 1]),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
