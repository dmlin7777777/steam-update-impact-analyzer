# agents/llm_review_agent.py
# LangGraph node: Claude haiku reviews flagged rows and decides keep/remove/uncertain.

from __future__ import annotations

import json

import anthropic

from agents.state import LLMReviewOutput, PipelineState, ReviewDecision
from core.features import extract_features
from config import LLM_MODEL_LIGHT

_client = anthropic.Anthropic()

_SYSTEM = """You are a Steam review data-quality auditor.
You receive a numbered list of reviews that were flagged as suspicious by automated rules
(too short, repeated characters, or high special-character ratio).

For each review decide:
- "keep"    → content is valid; just unusual formatting or very brief
- "remove"  → spam, gibberish, or completely meaningless
- "uncertain" → genuinely ambiguous; flag for human review

Reply with a JSON object exactly like:
{"results": [{"index": 0, "decision": "keep", "reason": "brief but genuine"}, ...]}
"""


def llm_review_node(state: PipelineState) -> dict:
    """
    Reads:  flagged_reviews, cleaned_reviews
    Writes: cleaned_reviews (merged with kept rows), llm_review_log, current_step

    Sends up to 50 flagged reviews to Claude haiku; merges "keep" rows back
    into cleaned_reviews and runs feature extraction on them.
    """
    flagged = state.get("flagged_reviews")
    cleaned = state.get("cleaned_reviews")

    if flagged is None or flagged.empty:
        return {"current_step": "llm_review_done"}

    # Cap at 50 rows per call to keep latency reasonable
    sample = flagged.head(50)
    numbered = "\n".join(
        f"[{i}] {row['review_content'][:300]}"
        for i, (_, row) in enumerate(sample.iterrows())
    )

    try:
        msg = _client.messages.create(
            model=LLM_MODEL_LIGHT,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": f"Review the following:\n{numbered}"}],
        )
        raw     = msg.content[0].text.strip()
        parsed  = json.loads(raw)
        results = parsed.get("results", [])
    except Exception as exc:
        return {
            "current_step": "llm_review_done",
            "errors":       [f"[llm_review] API call failed: {exc}"],
        }

    decisions = []
    keep_indices: set[int] = set()

    for item in results:
        idx      = item.get("index", -1)
        decision = item.get("decision", "uncertain")
        reason   = item.get("reason", "")

        if decision not in ("keep", "remove", "uncertain"):
            decision = "uncertain"

        decisions.append(ReviewDecision(index=idx, decision=decision, reason=reason))
        if decision == "keep":
            keep_indices.add(idx)

    # Merge kept rows (with features) back into cleaned_reviews
    if keep_indices:
        rows_to_keep = sample.iloc[sorted(keep_indices)].copy()
        rows_to_keep = extract_features(rows_to_keep)

        import pandas as pd
        merged = pd.concat(
            [df for df in [cleaned, rows_to_keep] if df is not None and not df.empty],
            ignore_index=True,
        )
    else:
        merged = cleaned

    output = LLMReviewOutput(
        decisions=  decisions,
        n_kept=     len(keep_indices),
        n_removed=  sum(1 for d in decisions if d.decision == "remove"),
        n_uncertain=sum(1 for d in decisions if d.decision == "uncertain"),
    )

    return {
        "cleaned_reviews": merged,
        "llm_review_log":  [output],       # Annotated[list, add] → appended
        "current_step":    "llm_review_done",
    }
