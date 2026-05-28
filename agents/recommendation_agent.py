# agents/recommendation_agent.py
# LangGraph node: Claude sonnet synthesises analysis + patch notes into
# a structured recommendation report.

from __future__ import annotations

import json

import anthropic

from agents.state import (
    PipelineState,
    Recommendations,
    RecommendationItem,
)
from config import LLM_MODEL

_client = anthropic.Anthropic()

_SYSTEM = """You are a senior game analytics consultant specialising in post-update player sentiment.

You will receive:
1. Pre-update and post-update sentiment statistics
2. Detected topic breakdown (what players are talking about)
3. Risk signals triggered by the update
4. Patch notes for the update (may be empty)

Your task:
A) Write a 2-3 paragraph executive summary in plain English explaining what happened
   after the update and why player sentiment shifted (or didn't).
B) Produce a prioritised list of concrete, actionable recommendations.

Reply with a single JSON object:
{
  "narrative": "...",
  "items": [
    {
      "priority": "P1",
      "issue": "...",
      "action": "...",
      "expected_impact": "..."
    }
  ]
}

Priority: P1 = urgent (fix within 48 h), P2 = important (within 1 week), P3 = nice-to-have.
Keep narrative under 300 words. Include 3-6 recommendation items.
"""


def recommendation_node(state: PipelineState) -> dict:
    """
    Reads:  analysis, patch_notes, game_name, event_comments
    Writes: recommendations, current_step
    """
    analysis      = state.get("analysis")
    patch_notes   = state.get("patch_notes", [])
    game_name     = state.get("game_name", "Unknown Game")
    event_df      = state.get("event_comments")

    if analysis is None:
        return {
            "current_step": "done",
            "errors":       ["[recommendation] no analysis available — skipping"],
        }

    # Build context string for Claude
    pre  = analysis.pre_sentiment
    post = analysis.post_sentiment

    context_parts = [
        f"Game: {game_name}",
        "",
        "=== SENTIMENT COMPARISON ===",
        f"Pre-update  : compound={pre.mean_compound:+.3f}  "
        f"pos={pre.positive_pct:.1%}  neu={pre.neutral_pct:.1%}  neg={pre.negative_pct:.1%}  "
        f"(n={pre.n_reviews})",
        f"Post-update : compound={post.mean_compound:+.3f}  "
        f"pos={post.positive_pct:.1%}  neu={post.neutral_pct:.1%}  neg={post.negative_pct:.1%}  "
        f"(n={post.n_reviews})",
        f"Delta       : {analysis.sentiment_delta:+.3f}",
        f"Overall risk: {analysis.overall_risk}",
        "",
        "=== TOPIC BREAKDOWN (post-update) ===",
    ]

    for tc in analysis.top_topics[:6]:
        context_parts.append(f"  {tc.label:<20} {tc.pct:.1%}  ({tc.count} reviews)")

    if analysis.risk_signals:
        context_parts += ["", "=== RISK SIGNALS ==="]
        for rs in analysis.risk_signals:
            context_parts.append(f"  [{rs.severity}] {rs.rule}: {rs.description}")

    if patch_notes:
        context_parts += ["", "=== PATCH NOTES ==="]
        for pn in patch_notes[:3]:    # top 3 most recent
            context_parts.append(f"--- {pn.title} ({pn.published_at.date()}) ---")
            context_parts.append(pn.contents[:800])
    else:
        context_parts += ["", "=== PATCH NOTES ===", "(none available)"]

    # Event comments — direct player reactions under the update announcement.
    # These are higher-signal than general reviews and should be weighted accordingly.
    if event_df is not None and not event_df.empty:
        import pandas as pd
        n_ec = len(event_df)
        context_parts += ["", f"=== ANNOUNCEMENT PAGE COMMENTS ({n_ec} total) ==="]
        context_parts.append(
            "These comments were posted directly under the update announcement "
            "and represent the most direct player reaction to this specific update."
        )
        # Sample up to 10 comments, prefer those with more upvotes
        sample = event_df.copy()
        if "votes_up" in sample.columns:
            sample = sample.sort_values("votes_up", ascending=False)
        for _, row in sample.head(10).iterrows():
            text    = str(row.get("review_content", "")).strip()[:200]
            upvotes = int(row.get("votes_up", 0))
            up_str  = f"  [{upvotes}👍]" if upvotes > 0 else ""
            context_parts.append(f"  •{up_str} {text}")
    else:
        context_parts += ["", "=== ANNOUNCEMENT PAGE COMMENTS ===",
                          "(none available — game may use JS-rendered comments)"]

    context = "\n".join(context_parts)

    try:
        msg = _client.messages.create(
            model=LLM_MODEL,
            max_tokens=1500,
            system=[{"type": "text", "text": _SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": context}],
        )
        raw    = msg.content[0].text.strip()
        parsed = json.loads(raw)

        items = [
            RecommendationItem(
                priority=        item["priority"],
                issue=           item["issue"],
                action=          item["action"],
                expected_impact= item["expected_impact"],
            )
            for item in parsed.get("items", [])
        ]

        recommendations = Recommendations(
            narrative= parsed.get("narrative", ""),
            items=     items,
        )

    except Exception as exc:
        # Graceful degradation — return a minimal report
        recommendations = Recommendations(
            narrative=f"Analysis complete. Risk level: {analysis.overall_risk}. "
                      f"Sentiment delta: {analysis.sentiment_delta:+.3f}. "
                      f"(Report generation failed: {exc})",
            items=[],
        )

    return {
        "recommendations": recommendations,
        "current_step":    "done",
    }
