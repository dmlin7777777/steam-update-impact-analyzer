# tests/test_e2e.py
# End-to-end smoke test using mocked Steam API calls.
# Does NOT hit the real Steam API or the Anthropic API.

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.state import PipelineState


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_review(text: str, days_offset: int, voted_up: bool = True) -> dict:
    """Helper to build a fake Steam review dict."""
    ts = datetime(2024, 3, 20, tzinfo=timezone.utc)
    import datetime as dt
    ts = ts + dt.timedelta(days=days_offset)
    return {
        "recommendationid": f"id_{abs(hash(text)) % 10000}",
        "review":           text,
        "voted_up":         voted_up,
        "timestamp_created": int(ts.timestamp()),
        "author":           {"playtime_forever": 120},
        "votes_up":         2,
        "votes_funny":      0,
    }


_FAKE_REVIEWS_API = {
    "success": 1,
    "reviews": [
        _make_review("Great game, loving the new patch content!", 1),
        _make_review("Performance is terrible after the update. FPS drops everywhere.", 1, False),
        _make_review("ok", 1),   # will be flagged (short)
        _make_review("The new gameplay mechanics are interesting but buggy.", 2),
        _make_review("Crashed three times already. Please fix this ASAP.", 2, False),
        _make_review("Amazing graphics update, well done devs.", 3),
    ],
    "cursor": "",
}

_FAKE_APP_DETAILS = {
    "730": {
        "success": True,
        "data": {
            "name": "Test Game",
            "developers": ["Test Dev"],
            "header_image": "",
        },
    }
}

_FAKE_NEWS = {
    "appnews": {
        "newsitems": [
            {
                "gid": "1",
                "title": "Patch 2.1 — Bug fixes and performance improvements",
                "url": "https://store.steampowered.com/news/1",
                "date": int(datetime(2024, 3, 19, tzinfo=timezone.utc).timestamp()),
                "contents": "We fixed a critical performance bug affecting all users.",
                "tags": [{"tag": "patchnotes"}],
            }
        ]
    }
}

_FAKE_LLM_REVIEW = json.dumps({
    "results": [{"index": 0, "decision": "remove", "reason": "too short"}]
})

_FAKE_LLM_TOPICS = json.dumps(["gameplay", "bugs", "performance"])

_FAKE_LLM_RECO = json.dumps({
    "narrative": "The update caused performance issues that led to negative sentiment.",
    "items": [
        {
            "priority": "P1",
            "issue": "FPS drops reported widely",
            "action": "Release hotfix targeting GPU optimisation",
            "expected_impact": "Reduce negative reviews by ~30%",
        }
    ],
})


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("agents.scraper_agent.fetch_event_comments")
@patch("core.scraper.requests.get")
@patch("anthropic.Anthropic")
def test_pipeline_runs_end_to_end(mock_anthropic_cls, mock_get, mock_fetch_ec):
    """Full graph smoke test with mocked Steam API and Anthropic API."""
    from graph import app

    # --- Mock Steam API -------------------------------------------------
    # NOTE: Do NOT also patch core.event_comments.requests.get here.
    # Both core.scraper and core.event_comments import the same 'requests'
    # module object, so a second patch on requests.get would overwrite the
    # first, causing _get() to use the wrong mock.  We mock fetch_event_comments
    # at the function level instead to avoid the conflict.
    def fake_get(url, params=None, timeout=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "appdetails" in url:
            resp.json.return_value = _FAKE_APP_DETAILS
        elif "ISteamNews" in url:
            resp.json.return_value = _FAKE_NEWS
        else:
            resp.json.return_value = _FAKE_REVIEWS_API
        return resp

    mock_get.side_effect = fake_get
    # Return zero event comments (graceful empty-list path)
    mock_fetch_ec.return_value = []

    # --- Mock Anthropic client ------------------------------------------
    def _make_content(text):
        content_block = MagicMock()
        content_block.text = text
        return [content_block]

    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    call_count = [0]
    def fake_create(**kwargs):
        msg = MagicMock()
        n = call_count[0]
        call_count[0] += 1
        # Route by expected output shape
        if n == 0:
            msg.content = _make_content(_FAKE_LLM_REVIEW)   # llm_review
        elif n == 1:
            msg.content = _make_content(_FAKE_LLM_TOPICS)   # topic tagging
        else:
            msg.content = _make_content(_FAKE_LLM_RECO)     # recommendation
        return msg

    mock_client.messages.create.side_effect = fake_create

    # --- Run graph ------------------------------------------------------
    initial: PipelineState = {
        "appid":          "730",
        "update_date":    datetime(2024, 3, 20, tzinfo=timezone.utc),
        "pre_days":       7,
        "post_days":      7,
        "game_name":      "",
        "pre_reviews":    None,
        "post_reviews":   None,
        "patch_notes":    [],
        "event_comments": None,
        "cleaned_reviews": None,
        "flagged_reviews": None,
        "analysis":       None,
        "llm_review_log": [],
        "recommendations": None,
        "current_step":   "start",
        "errors":         [],
    }

    result = app.invoke(initial)

    # --- Assertions -----------------------------------------------------
    assert result["current_step"] == "done", f"Unexpected step: {result['current_step']}"
    assert result["game_name"] == "Test Game"
    assert result["analysis"] is not None
    assert result["recommendations"] is not None
    assert result["recommendations"].narrative != ""


def test_feature_extraction_columns():
    """Unit test: extract_features adds expected columns."""
    from core.features import extract_features

    df = pd.DataFrame({
        "review_content": ["Great game!", "Absolute garbage, broken update."],
        "voted_up":       [True, False],
    })
    df = extract_features(df)
    expected = {
        "review_content_processed", "vader_compound", "vader_pos",
        "sentiment_label", "word_count", "flesch_reading_ease", "is_recommended",
    }
    assert expected.issubset(set(df.columns)), f"Missing: {expected - set(df.columns)}"


def test_cleaning_flags_short_text():
    """Unit test: run_cleaning flags reviews shorter than 10 chars."""
    from core.cleaning import run_cleaning

    df = pd.DataFrame({
        "review_content": ["ok", "This is a wonderful and detailed game review."]
    })
    cleaned, flagged = run_cleaning(df)
    assert len(flagged) == 1
    assert len(cleaned) == 1
    assert flagged.iloc[0]["review_content"] == "ok"


def test_sentiment_stats_empty_df():
    """Unit test: compute_sentiment_stats handles empty DataFrame gracefully."""
    from core.analysis import compute_sentiment_stats

    stats = compute_sentiment_stats(pd.DataFrame())
    assert stats.n_reviews == 0
    assert stats.mean_compound == 0.0
