# core/event_analysis.py
# Map-Reduce LLM analysis of event comments (update announcement reactions).
#
# Why not VADER?
# --------------
# Event comments are the primary signal for update impact analysis. They are
# short, context-heavy, and full of sarcasm/gaming slang that VADER misjudges.
# They also lack a voted_up field, so there is no independent signal to
# cross-validate VADER against. The only reliable way to assess sentiment is
# to have the LLM read the raw text directly.
#
# Why Map-Reduce?
# ---------------
# A single update may have 500-2000+ comments. Sending all of them in one LLM
# call risks lost-in-the-middle attention decay. Map-Reduce ensures every
# comment gets full attention in its chunk, and structured intermediate output
# preserves information across the aggregation boundary.
#
# Architecture:
#   Map   : Chunk comments (~100 per chunk) -> V4 Flash reads each chunk ->
#           structured JSON (sentiment counts, themes, representative quotes)
#   Reduce: Aggregate structured outputs -> one V4 Pro call with aggregated
#           stats + representative quotes -> final qualitative assessment

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from config import LLM_MODEL, LLM_MODEL_LIGHT
from core.llm import chat as llm_chat

CHUNK_SIZE = 50   # comments per Map call (DeepSeek V4 Flash is reliable at 50,
                   # starts returning empty above ~70 with real-world comment lengths)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ChunkAnalysis:
    """Structured output from one Map call."""
    positive:    int = 0
    negative:    int = 0
    neutral:     int = 0
    themes:      list[dict] = field(default_factory=list)   # [{"label": ..., "count": ...}]
    quotes:      list[dict] = field(default_factory=list)   # [{"text": ..., "sentiment": ...}]
    total:       int = 0


@dataclass
class EventCommentAnalysis:
    """Final output of the full Map-Reduce pipeline."""
    n_comments:        int
    positive_count:    int
    negative_count:    int
    neutral_count:     int
    positive_pct:      float
    negative_pct:      float
    neutral_pct:       float
    top_themes:        list[dict]    # [{"label": ..., "count": ...}], sorted desc
    representative_quotes: list[dict]
    llm_summary:       str           # qualitative assessment from Reduce step
    patch_context:     str           # patch notes used as context


# ── Map phase ────────────────────────────────────────────────────────────────

_MAP_SYSTEM = """You are analysing player comments posted directly under a game update announcement on Steam.

For this batch of comments, produce a JSON object with exactly these keys:
{
  "sentiment": {"positive": <int>, "negative": <int>, "neutral": <int>},
  "themes": [{"label": "<short topic phrase>", "count": <int>}, ...],
  "quotes": [{"text": "<exact quote, max 120 chars>", "sentiment": "<positive|negative|neutral>"}, ...]
}

Rules:
- Classify EVERY comment's sentiment. positive + negative + neutral must equal the total comment count.
- Understand gaming slang and sarcasm: "L update", "dead game", "RIP" = negative. "W", "fire update" = positive.
- themes: identify 3-5 recurring topics in this batch. Use descriptive phrases, not single words.
- quotes: pick 2-4 comments that best represent the range of opinions. Copy the text exactly.
- Return ONLY the JSON object, no other text."""


def _strip_code_fence(raw: str) -> str:
    """Strip markdown code fences (```, ```json, etc.) from LLM output."""
    # Remove opening fence: ```json, ```JSON, ```, etc.
    raw = re.sub(r'^```(?:json|JSON)?\s*\n?', '', raw.strip())
    # Remove closing fence
    raw = re.sub(r'\n?```\s*$', '', raw)
    return raw.strip()


def _map_chunk(
    comments: list[str],
    patch_context: str,
    max_retries: int = 3,
) -> ChunkAnalysis:
    """Run one Map call on a chunk of comments with retry on parse failure."""
    numbered = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(comments))

    user_msg = f"""Update context (from patch notes):
{patch_context[:800]}

---
{len(comments)} player comments to analyse:
{numbered}"""

    for attempt in range(max_retries):
        try:
            raw = llm_chat(
                model=LLM_MODEL_LIGHT,
                system=_MAP_SYSTEM,
                user=user_msg,
                max_tokens=2048,
                json_mode=True,
            )
            if not raw or not raw.strip():
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return ChunkAnalysis(total=len(comments))

            raw = _strip_code_fence(raw)
            d = json.loads(raw)
            sent = d.get("sentiment", {})
            return ChunkAnalysis(
                positive=int(sent.get("positive", 0)),
                negative=int(sent.get("negative", 0)),
                neutral=int(sent.get("neutral", 0)),
                themes=d.get("themes", []),
                quotes=d.get("quotes", []),
                total=len(comments),
            )
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return ChunkAnalysis(total=len(comments))


# ── Aggregation ──────────────────────────────────────────────────────────────

def _aggregate_chunks(chunks: list[ChunkAnalysis]) -> dict:
    """Merge structured outputs from all Map calls."""
    total_pos = sum(c.positive for c in chunks)
    total_neg = sum(c.negative for c in chunks)
    total_neu = sum(c.neutral for c in chunks)
    total_n   = sum(c.total for c in chunks)

    # Merge themes: count how many chunks mention each theme label,
    # and sum up per-chunk counts.
    theme_counts: dict[str, int] = {}
    for c in chunks:
        for t in c.themes:
            label = t.get("label", "").strip().lower()
            if label:
                theme_counts[label] = theme_counts.get(label, 0) + int(t.get("count", 1))

    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    top_themes = [{"label": label, "count": count} for label, count in sorted_themes[:10]]

    # Collect all representative quotes, deduplicate by text
    seen_texts: set[str] = set()
    all_quotes: list[dict] = []
    for c in chunks:
        for q in c.quotes:
            text = q.get("text", "").strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                all_quotes.append(q)

    return {
        "n_comments":  total_n,
        "positive":    total_pos,
        "negative":    total_neg,
        "neutral":     total_neu,
        "top_themes":  top_themes,
        "quotes":      all_quotes,
    }


# ── Reduce phase ─────────────────────────────────────────────────────────────

_REDUCE_SYSTEM = """You are a senior game analyst writing an internal assessment of player reactions to a game update.

You will receive:
1. The patch notes for the update
2. Aggregated sentiment statistics from all comments under the announcement
3. The top recurring themes players discussed
4. Representative quotes from players

Write a concise assessment (150-250 words) that:
- States the overall player reception (positive / mixed / negative) with specific numbers
- Identifies the 2-3 most important player concerns or praises
- Notes any disconnect between what the developers highlighted and what players actually care about
- Flags any signals that require immediate developer attention

Be direct and specific. Use the quotes as evidence. Do not hedge or use corporate language.
Return ONLY the assessment text, no JSON."""


def _reduce(
    aggregated: dict,
    patch_context: str,
) -> str:
    """Synthesize aggregated Map outputs into a qualitative assessment."""
    total = aggregated["n_comments"]
    pos_pct = aggregated["positive"] / max(total, 1) * 100
    neg_pct = aggregated["negative"] / max(total, 1) * 100
    neu_pct = aggregated["neutral"]  / max(total, 1) * 100

    themes_str = "\n".join(
        f"  - {t['label']} ({t['count']} mentions)"
        for t in aggregated["top_themes"][:8]
    )

    quotes_str = "\n".join(
        f'  [{q.get("sentiment", "?")}] "{q["text"]}"'
        for q in aggregated["quotes"][:15]
    )

    user_msg = f"""=== PATCH NOTES ===
{patch_context[:1200]}

=== COMMENT STATISTICS ({total} total) ===
Positive: {aggregated['positive']} ({pos_pct:.1f}%)
Negative: {aggregated['negative']} ({neg_pct:.1f}%)
Neutral:  {aggregated['neutral']} ({neu_pct:.1f}%)

=== TOP THEMES ===
{themes_str}

=== REPRESENTATIVE QUOTES ===
{quotes_str}"""

    try:
        return llm_chat(
            model=LLM_MODEL,
            system=_REDUCE_SYSTEM,
            user=user_msg,
            max_tokens=600,
        )
    except Exception as exc:
        return (
            f"Reduce failed ({exc}). Raw stats: {total} comments, "
            f"{pos_pct:.0f}% positive, {neg_pct:.0f}% negative."
        )


# ── Public API ───────────────────────────────────────────────────────────────

def analyse_event_comments(
    comments: list[str],
    patch_notes_text: str,
    chunk_size: int = CHUNK_SIZE,
) -> Optional[EventCommentAnalysis]:
    """
    Full Map-Reduce analysis of event comments.

    Args:
        comments:         List of raw comment strings (no preprocessing).
        patch_notes_text: Patch notes for context (what the update changed).
        chunk_size:       Comments per Map call (default 100).

    Returns:
        EventCommentAnalysis with sentiment counts, themes, quotes,
        and a qualitative LLM summary. None if no comments provided.
    """
    if not comments:
        return None

    # ── Map ───────────────────────────────────────────────────────────────────
    n_chunks = math.ceil(len(comments) / chunk_size)
    chunks: list[ChunkAnalysis] = []

    for i in range(n_chunks):
        batch = comments[i * chunk_size : (i + 1) * chunk_size]
        result = _map_chunk(batch, patch_notes_text)
        chunks.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = _aggregate_chunks(chunks)

    # ── Reduce ────────────────────────────────────────────────────────────────
    summary = _reduce(agg, patch_notes_text)

    total = agg["n_comments"]
    return EventCommentAnalysis(
        n_comments=total,
        positive_count=agg["positive"],
        negative_count=agg["negative"],
        neutral_count=agg["neutral"],
        positive_pct=round(agg["positive"] / max(total, 1), 4),
        negative_pct=round(agg["negative"] / max(total, 1), 4),
        neutral_pct=round(agg["neutral"]   / max(total, 1), 4),
        top_themes=agg["top_themes"],
        representative_quotes=agg["quotes"][:10],
        llm_summary=summary,
        patch_context=patch_notes_text[:500],
    )
