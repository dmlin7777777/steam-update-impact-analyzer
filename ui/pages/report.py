# ui/pages/report.py
# Report tab: narrative summary + structured recommendations + LLM decision log.

from __future__ import annotations

import pandas as pd
import streamlit as st

from agents.state import LLMReviewOutput, Recommendations

_PRIORITY_COLOUR = {"P1": "#e74c3c", "P2": "#f39c12", "P3": "#27ae60"}
_PRIORITY_LABEL  = {"P1": "🔴 P1 Urgent", "P2": "🟡 P2 Important", "P3": "🟢 P3 Nice-to-have"}


def render_report(result: dict) -> None:
    """Render the analysis report from a completed pipeline result."""
    reco: Recommendations | None = result.get("recommendations")
    analysis = result.get("analysis")

    if reco is None:
        st.info("No report yet. Run the pipeline from the **Updates** tab.")
        return

    game_name   = result.get("game_name", "Unknown Game")
    update_date = result.get("update_date")

    # ── Header ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(f"📄 Post-Update Analysis Report")
        if update_date:
            st.caption(
                f"**{game_name}** · Update analysed: "
                f"{pd.Timestamp(update_date).strftime('%Y-%m-%d')}"
            )
    with col2:
        if analysis:
            risk = analysis.overall_risk
            badge = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
            st.metric("Risk", f"{badge.get(risk, '')} {risk}")

    st.divider()

    # ── Executive narrative ───────────────────────────────────────────────────
    st.subheader("Executive Summary")
    if reco.narrative:
        # Split by paragraph breaks for cleaner rendering
        for para in reco.narrative.split("\n\n"):
            para = para.strip()
            if para:
                st.write(para)
    else:
        st.info("No narrative generated.")

    st.divider()

    # ── Structured recommendations ────────────────────────────────────────────
    st.subheader("Action Items")

    if reco.items:
        # Group by priority
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
                        st.info(f"💡 {item.expected_impact}")

            st.write("")  # spacer between priority groups
    else:
        st.success("No immediate action items identified.")

    st.divider()

    # ── LLM decision log (collapsible) ────────────────────────────────────────
    st.subheader("LLM Review Log")
    llm_log: list[LLMReviewOutput] = result.get("llm_review_log", [])

    if not llm_log:
        st.caption("No LLM cleaning review was triggered (flagged ratio below threshold).")
    else:
        for i, entry in enumerate(llm_log):
            with st.expander(
                f"Review pass #{i + 1} — kept {entry.n_kept}, "
                f"removed {entry.n_removed}, uncertain {entry.n_uncertain}"
            ):
                rows = [
                    {
                        "Index":    d.index,
                        "Decision": d.decision.upper(),
                        "Reason":   d.reason,
                    }
                    for d in entry.decisions
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Raw analysis numbers (debug/transparency) ─────────────────────────────
    if analysis:
        with st.expander("📊 Raw Analysis Numbers"):
            pre  = analysis.pre_sentiment
            post = analysis.post_sentiment
            st.write("**Pre-update sentiment**")
            st.json(pre.model_dump())
            st.write("**Post-update sentiment**")
            st.json(post.model_dump())
            st.write("**Risk signals**")
            st.json([s.model_dump() for s in analysis.risk_signals])
