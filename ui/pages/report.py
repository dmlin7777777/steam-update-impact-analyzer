# ui/pages/report.py
# Report tab: narrative summary + structured recommendations.

from __future__ import annotations

import pandas as pd
import streamlit as st

from agents.state import Recommendations

_PRIORITY_COLOUR = {"P1": "#e74c3c", "P2": "#f39c12", "P3": "#27ae60"}
_PRIORITY_LABEL  = {"P1": "P1 Urgent", "P2": "P2 Important", "P3": "P3 Nice-to-have"}


def render_report(result: dict) -> None:
    """Render the analysis report from a completed pipeline result."""
    reco: Recommendations | None = result.get("recommendations")
    analysis = result.get("analysis")

    if reco is None:
        st.info("No report yet. Run the pipeline from the **Updates** tab.")
        return

    game_name   = result.get("game_name", "Unknown Game")
    update_date = result.get("update_date")

    # ── Header ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("Post-Update Analysis Report")
        if update_date:
            st.caption(
                f"**{game_name}** | Update analysed: "
                f"{pd.Timestamp(update_date).strftime('%Y-%m-%d')}"
            )
    with col2:
        if analysis:
            st.metric("Risk", analysis.overall_risk)

    st.divider()

    # ── Executive narrative ──────────────────────────────────────────────────
    st.subheader("Executive Summary")
    if reco.narrative:
        for para in reco.narrative.split("\n\n"):
            para = para.strip()
            if para:
                st.write(para)
    else:
        st.info("No narrative generated.")

    st.divider()

    # ── Structured recommendations ───────────────────────────────────────────
    st.subheader("Action Items")

    if reco.items:
        for priority in ("P1", "P2", "P3"):
            group = [item for item in reco.items if item.priority == priority]
            if not group:
                continue
            st.markdown(
                f"<span style='color:{_PRIORITY_COLOUR[priority]};font-weight:bold'>"
                f"{_PRIORITY_LABEL[priority]}</span>",
                unsafe_allow_html=True,
            )
            for item in group:
                with st.container(border=True):
                    c1, c2 = st.columns([5, 3])
                    with c1:
                        st.markdown(f"**Issue:** {item.issue}")
                        st.markdown(f"**Action:** {item.action}")
                    with c2:
                        st.info(item.expected_impact)
            st.write("")
    else:
        st.success("No immediate action items identified.")

    st.divider()

    # ── Raw analysis numbers (debug/transparency) ────────────────────────────
    if analysis:
        with st.expander("Raw Analysis Numbers"):
            st.write("**Pre-update sentiment (VADER)**")
            st.json(analysis.pre_sentiment.model_dump())

            if analysis.review_analysis:
                st.write("**Post-update review sentiment (VADER)**")
                st.json(analysis.review_analysis.sentiment.model_dump())
                if analysis.review_analysis.n_disagreements > 0:
                    st.write(
                        f"**VADER disagreements:** {analysis.review_analysis.n_disagreements} "
                        f"({analysis.review_analysis.disagreement_pct:.1%})"
                    )

            if analysis.event_analysis:
                st.write(f"**Event comment analysis (LLM, {analysis.event_analysis.n_comments} comments)**")
                st.json({
                    "positive_pct": analysis.event_analysis.positive_pct,
                    "negative_pct": analysis.event_analysis.negative_pct,
                    "neutral_pct":  analysis.event_analysis.neutral_pct,
                    "top_themes":   analysis.event_analysis.top_themes[:5],
                })

            st.write("**Risk signals**")
            st.json([s.model_dump() for s in analysis.risk_signals])
