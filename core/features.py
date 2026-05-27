# core/features.py
# General-purpose NLP feature extraction — no GPU, no genre-specific training.
#
# Adds the following columns to a review DataFrame:
#   review_content_processed  str    URL/HTML-stripped, lowercased text
#   vader_compound            float  VADER compound score  [-1, 1]
#   vader_pos                 float  VADER positive ratio
#   vader_neu                 float  VADER neutral  ratio
#   vader_neg                 float  VADER negative ratio
#   sentiment_label           str    "positive" | "neutral" | "negative"
#   char_count                int    character count of processed text
#   word_count                int    whitespace-split word count
#   avg_word_length           float  mean characters per word
#   flesch_reading_ease       float  Flesch readability score (0–100)
#   is_recommended            bool   copy of voted_up (convenience alias)

from __future__ import annotations

import html
import re

import pandas as pd
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import VADER_GRAY_LO

# Shared VADER analyser (thread-safe for read-only calls)
_vader = SentimentIntensityAnalyzer()

# ── Text pre-processing ───────────────────────────────────────────────────────

_URL_RE  = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE   = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Strip HTML entities, URLs, tags; collapse whitespace; lowercase."""
    text = html.unescape(str(text))
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.lower()


# ── VADER sentiment ───────────────────────────────────────────────────────────

def _vader_scores(text: str) -> dict:
    return _vader.polarity_scores(text)


def _sentiment_label(compound: float) -> str:
    if compound >= VADER_GRAY_LO:
        return "positive"
    if compound <= -VADER_GRAY_LO:
        return "negative"
    return "neutral"


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add NLP feature columns to a review DataFrame in-place (returns same df).

    Requires column: review_content
    Optional column: voted_up (copied to is_recommended if present)

    Idempotent — skips columns that already exist.
    """
    df = df.copy()

    # 1. Text pre-processing
    if "review_content_processed" not in df.columns:
        df["review_content_processed"] = df["review_content"].apply(_clean_text)

    processed = df["review_content_processed"]

    # 2. VADER sentiment (vectorised row-by-row; VADER is fast enough)
    if "vader_compound" not in df.columns:
        scores = processed.apply(_vader_scores)
        df["vader_compound"] = scores.apply(lambda s: s["compound"])
        df["vader_pos"]      = scores.apply(lambda s: s["pos"])
        df["vader_neu"]      = scores.apply(lambda s: s["neu"])
        df["vader_neg"]      = scores.apply(lambda s: s["neg"])
        df["sentiment_label"] = df["vader_compound"].apply(_sentiment_label)

    # 3. Basic text statistics
    if "char_count" not in df.columns:
        df["char_count"] = processed.str.len()

    if "word_count" not in df.columns:
        words = processed.str.split()
        df["word_count"]     = words.apply(len)
        df["avg_word_length"] = words.apply(
            lambda ws: sum(len(w) for w in ws) / len(ws) if ws else 0.0
        )

    # 4. Readability (textstat handles empty strings gracefully)
    if "flesch_reading_ease" not in df.columns:
        df["flesch_reading_ease"] = processed.apply(textstat.flesch_reading_ease)

    # 5. Convenience alias for voted_up
    if "is_recommended" not in df.columns and "voted_up" in df.columns:
        df["is_recommended"] = df["voted_up"].astype(bool)

    return df


def is_gray_zone(compound: float) -> bool:
    """True when VADER is ambiguous and a Claude re-score would help."""
    return -VADER_GRAY_LO < compound < VADER_GRAY_LO
