# agents/recommendation_agent.py
# LangGraph node: Claude sonnet synthesises two independent analysis paths
# into a structured recommendation report.

from __future__ import annotations

import json

from agents.state import (
    PipelineState,
    Recommendations,
    RecommendationItem,
)
from core.llm import chat as llm_chat
from config import LLM_MODEL

_SYSTEM = """You are a senior game analytics consultant specialising in post-update player sentiment.

You will receive TWO independent data sources:

1. ANNOUNCEMENT COMMENTS (primary signal):
   Comments posted directly under the update announcement on Steam.
   These are analysed by an LLM that read every comment. The sentiment stats,
   themes, and representative quotes come directly from reading the raw text.

2. GENERAL REVIEWS (secondary signal):
   Reviews from the Steam store page posted in the same time window.
   These are analysed with VADER (rule-based sentiment) + disagreement detection.
   VADER can misclassify sarcasm — disagreement stats show where it likely failed.

When the two sources disagree, trust the announcement comments more — they are
direct reactions to THIS specific update, while reviews may discuss the game in general.

Your task:
A) Write a 2-3 paragraph executive summary explaining what happened after the update.
   Use specific numbers and quotes as evidence. Be direct — no hedging.
B) Produce a prioritised list of concrete, actionable recommendations.

Reply with a single JSON object:
{
  "narrative": "...",
  "items": [
    {"priority": "P1", "issue": "...", "action": "...", "expected_impact": "..."}
  ]
}

Priority: P1 = urgent (fix within 48 h), P2 = important (within 1 week), P3 = nice-to-have.
Keep narrative under 300 words. Include 3-6 recommendation items.
"""


def recommendation_node(state: PipelineState) -> dict:
    """
    Reads:  analysis, patch_notes, game_name
    Writes: recommendations, current_step
    """
    analysis    = state.get("analysis")
    patch_notes = state.get("patch_notes", [])
    game_name   = state.get("game_name", "Unknown Game")

    if analysis is None:
        return {
            "current_step": "done",
            "errors":       ["WARN: [recommendation] no analysis available -- skipping"],
        }

    pre = analysis.pre_sentiment

    context_parts = [
        f"Game: {game_name}",
        f"Overall risk: {analysis.overall_risk}",
        "",
    ]

    # ── Baseline ─────────────────────────────────────────────────────────────
    context_parts += [
        "=== BASELINE (pre-update reviews) ===",
        f"pos={pre.positive_pct:.1%}  neu={pre.neutral_pct:.1%}  neg={pre.negative_pct:.1%}  "
        f"(n={pre.n_reviews})",
        "",
    ]

    # ── Source 1: Announcement comments (primary) ────────────────────────────
    ea = analysis.event_analysis
    if ea is not None:
        context_parts += [
            f"=== ANNOUNCEMENT COMMENTS — PRIMARY ({ea.n_comments} comments, LLM-analysed) ===",
            f"Sentiment: {ea.positive_pct:.0%} positive, "
            f"{ea.negative_pct:.0%} negative, {ea.neutral_pct:.0%} neutral",
            "",
            "Top themes:",
        ]
        for t in ea.top_themes[:6]:
            context_parts.append(f"  - {t['label']} ({t['count']} mentions)")

        context_parts += ["", "Representative quotes:"]
        for q in ea.representative_quotes[:10]:
            sent = q.get("sentiment", "?")
            context_parts.append(f'  [{sent}] "{q["text"]}"')

        context_parts += ["", "LLM assessment:", ea.llm_summary, ""]
    else:
        context_parts += [
            "=== ANNOUNCEMENT COMMENTS ===",
            "(none available -- game may use JS-rendered comments)",
            "",
        ]

    # ── Source 2: General reviews (secondary) ────────────────────────────────
    ra = analysis.review_analysis
    if ra is not None:
        s = ra.sentiment
        context_parts += [
            f"=== GENERAL REVIEWS — SECONDARY ({s.n_reviews} reviews, VADER-scored) ===",
            f"pos={s.positive_pct:.1%}  neu={s.neutral_pct:.1%}  neg={s.negative_pct:.1%}",
            f"Negative ratio change vs baseline: {analysis.sentiment_delta:+.1%}",
        ]
        if ra.n_disagreements > 0:
            context_parts.append(
                f"VADER disagreements: {ra.n_disagreements} reviews "
                f"({ra.disagreement_pct:.1%}) where VADER sentiment contradicts "
                f"player's own thumbs-up/down vote (likely sarcasm/irony)"
            )
        if ra.distribution_summary:
            context_parts += ["", "Data distribution:", ra.distribution_summary]
        context_parts.append("")
    else:
        context_parts += [
            "=== GENERAL REVIEWS ===",
            "(no post-update reviews available)",
            "",
        ]

    # ── Topics ───────────────────────────────────────────────────────────────
    if analysis.top_topics:
        context_parts += ["=== TOPIC BREAKDOWN ==="]
        for tc in analysis.top_topics[:8]:
            context_parts.append(f"  {tc.label:<30} ({tc.count} mentions)")
        context_parts.append("")

    # ── Risk signals ─────────────────────────────────────────────────────────
    if analysis.risk_signals:
        context_parts += ["=== RISK SIGNALS ==="]
        for rs in analysis.risk_signals:
            context_parts.append(f"  [{rs.severity}] {rs.rule}: {rs.description}")
        context_parts.append("")

    # ── Patch notes ──────────────────────────────────────────────────────────
    if patch_notes:
        context_parts += ["=== PATCH NOTES ==="]
        for pn in patch_notes[:3]:
            context_parts.append(f"--- {pn.title} ({pn.published_at.date()}) ---")
            context_parts.append(pn.contents[:800])
    else:
        context_parts += ["=== PATCH NOTES ===", "(none available)"]

    context = "\n".join(context_parts)

    try:
        raw = llm_chat(
            model=LLM_MODEL,
            system=_SYSTEM,
            user=context,
            max_tokens=4096,
            json_mode=True,
        )
        parsed = json.loads(raw)
        items = [
            RecommendationItem(
                priority=item["priority"],
                issue=item["issue"],
                action=item["action"],
                expected_impact=item["expected_impact"],
            )
            for item in parsed.get("items", [])
        ]
        recommendations = Recommendations(
            narrative=parsed.get("narrative", ""),
            items=items,
        )
    except Exception as exc:
        recommendations = Recommendations(
            narrative=f"Analysis complete. Risk level: {analysis.overall_risk}. "
                      f"Negative ratio change: {analysis.sentiment_delta:+.1%}. "
                      f"(Report generation failed: {exc})",
            items=[],
        )

    return {
        "recommendations": recommendations,
        "current_step":    "done",
    }
