# agents/cleaning_agent.py
# LangGraph node: exclude non-analysable and spam/bot reviews, run feature extraction.
#
# Cleaning is deterministic — no LLM call. Rules are data-driven
# (validated on 59k CS2 reviews) and literature-backed.
# See core/cleaning.py for rule details and references.

from __future__ import annotations

from agents.state import PipelineState
from core.cleaning import run_cleaning
from core.features import extract_features


def cleaning_node(state: PipelineState) -> dict:
    """
    Reads:  post_reviews
    Writes: cleaned_reviews, current_step

    Two-tier exclusion (see core/cleaning.py):
      1. Hard exclude: null, non-English, punctuation-only
      2. Flagged exclude: behavioral spam/bot signals, content anomalies

    Runs feature extraction on surviving reviews so downstream agents
    have vader_compound, sentiment_label, etc. immediately.
    """
    df = state.get("post_reviews")

    if df is None or df.empty:
        return {
            "cleaned_reviews": df,
            "current_step":   "cleaning_done",
            "errors":         ["WARN: [cleaning] post_reviews is empty — nothing to clean"],
        }

    cleaned, excluded = run_cleaning(df)

    # Add NLP features to cleaned reviews right away
    if not cleaned.empty:
        cleaned = extract_features(cleaned)

    errors: list[str] = []
    if len(excluded) > 0:
        # Summarize exclusion reasons
        reason_counts = excluded["exclude_reason"].value_counts()
        breakdown = ", ".join(f"{reason}={count}" for reason, count in reason_counts.items())
        errors.append(
            f"INFO: [cleaning] Excluded {len(excluded)}/{len(excluded)+len(cleaned)} "
            f"reviews ({breakdown})"
        )

    return {
        "cleaned_reviews": cleaned,
        "current_step":   "cleaning_done",
        **({"errors": errors} if errors else {}),
    }
