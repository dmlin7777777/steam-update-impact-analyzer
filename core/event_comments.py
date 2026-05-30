# core/event_comments.py
# Fetch player comments posted directly under Steam update announcements.
#
# Why event comments matter
# -------------------------
# Review windows (pre/post) capture general sentiment over a time range.
# Event comments are *anchored to the specific announcement* — they are
# the most direct, unambiguous signal of how players reacted to one update.
# They should be treated as a high-weight supplement to the review windows.
#
# Steam event comment formats
# ---------------------------
# Steam has two event-page layouts depending on when the game/event was created:
#
# Format A (older, ≤ ~2024):
#   URL  : https://steamcommunity.com/app/{appid}/eventcomments/{event_gid}
#   HTML : comments embedded in initial HTML as .commentthread_comment nodes
#   Parse: BeautifulSoup, no auth required, pagination via ?ctp=N
#
# Format B (newer, 2025+):
#   URL  : https://steamcommunity.com/games/{slug}/announcements/detail/{gid}
#   HTML : skeleton page; comments loaded by JavaScript after page load
#   Parse: HTML parsing returns 0 comments → graceful empty return
#
# This module attempts Format A first (via the event GID from the resolved URL),
# then falls back silently.  Format B is tracked in the TODO below.
#
# TODO: add optional Playwright/Selenium path for Format B games.

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
_DELAY = 0.5         # seconds between paginated requests
_MAX_PAGES = 20      # cap at 20 pages × ~15 comments = ~300 comments


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_event_comments(
    appid: str,
    news_url: str,
    news_gid: str,
    max_pages: int = _MAX_PAGES,
) -> list[dict]:
    """
    Fetch player comments posted under a Steam update announcement.

    Strategy:
      1. Follow *news_url* redirect → extract the community event GID.
      2. Try Format A:  /app/{appid}/eventcomments/{event_gid}?ctp=N
      3. If Format A returns 0 comments, also try with *news_gid* as GID.
      4. If nothing works, return [] — caller should proceed without comments.

    Args:
        appid:    Steam App ID string.
        news_url: URL from GetNewsForApp (may redirect to Steam Community).
        news_gid: GID from GetNewsForApp (used as fallback GID).
        max_pages: Maximum pages to paginate (each ~15 comments).

    Returns:
        List of dicts with keys: comment_id, author, timestamp_raw,
        published_at (UTC datetime), content, upvotes.
        Empty list on any failure or when format is unsupported.
    """
    # Step 1: resolve the actual community event GID
    event_gid = _resolve_event_gid(news_url) or news_gid

    # Step 2: try the known-working Format A endpoint
    comments = _scrape_format_a(appid, event_gid, max_pages)

    # Step 3: if event_gid differed from news_gid and we got nothing, try news_gid
    if not comments and event_gid != news_gid:
        comments = _scrape_format_a(appid, news_gid, max_pages)

    return comments


def event_comments_available(appid: str, news_url: str, news_gid: str) -> bool:
    """Quick probe: returns True if at least one comment page can be fetched."""
    event_gid = _resolve_event_gid(news_url) or news_gid
    page = _fetch_comment_page(appid, event_gid, 1)
    return len(page) > 0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_event_gid(news_url: str) -> Optional[str]:
    """
    Follow *news_url* and extract the event GID from the resolved URL.
    Returns None if the URL doesn't resolve to a Steam Community event page.
    """
    try:
        r = requests.get(news_url, headers=_HEADERS, timeout=10, allow_redirects=True)
        # Match /eventcomments/{gid} or /announcements/detail/{gid}
        m = re.search(r"/(?:eventcomments|announcements/detail)/(\d+)", r.url)
        return m.group(1) if m else None
    except Exception:
        return None


def _scrape_format_a(appid: str, event_gid: str, max_pages: int) -> list[dict]:
    """
    Paginate through /app/{appid}/eventcomments/{event_gid}?ctp=N.
    Returns list of parsed comment dicts (may be empty).
    """
    all_comments: list[dict] = []
    for page in range(1, max_pages + 1):
        batch = _fetch_comment_page(appid, event_gid, page)
        if not batch:
            break
        all_comments.extend(batch)
        if len(batch) < 14:  # last page is usually < full size
            break
        time.sleep(_DELAY)
    return all_comments


def _fetch_comment_page(appid: str, event_gid: str, page: int) -> list[dict]:
    """
    Fetch one page of comments. Returns [] if unavailable or format is B.
    """
    url = (
        f"https://steamcommunity.com/app/{appid}"
        f"/eventcomments/{event_gid}?ctp={page}"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=10, allow_redirects=True)
    except Exception:
        return []

    # If Steam redirected away (dropped the GID → format B or not found)
    if event_gid not in r.url.split("?")[0]:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    comments = soup.select(".commentthread_comment")

    result: list[dict] = []
    for node in comments:
        author_el = node.select_one(".commentthread_author_link")
        ts_el     = node.select_one(".commentthread_comment_timestamp")
        text_el   = node.select_one(".commentthread_comment_text")
        up_el     = node.select_one(".commentthread_comment_upvotes, .upvote_count")

        # Parse timestamp — Steam formats vary by locale; store raw + attempt UTC
        ts_raw = ts_el.get_text(strip=True) if ts_el else ""
        pub_at = _parse_steam_timestamp(ts_raw)

        # Upvote count may be inside a data attribute or text
        upvotes = 0
        if up_el:
            try:
                upvotes = int(re.sub(r"[^\d]", "", up_el.get_text(strip=True)) or "0")
            except ValueError:
                upvotes = 0

        result.append({
            "comment_id":    node.get("id", ""),
            "author":        author_el.get_text(strip=True) if author_el else "",
            "timestamp_raw": ts_raw,
            "published_at":  pub_at,
            "content":       text_el.get_text(strip=True) if text_el else "",
            "upvotes":       upvotes,
        })

    return result


def _parse_steam_timestamp(ts: str) -> Optional[datetime]:
    """
    Attempt to parse Steam's locale-specific timestamp string.
    Returns UTC datetime or None if parsing fails.

    Steam renders timestamps like:
      "Nov 7, 2024 @ 11:06pm"           (English)
      "2024 年 11 月 7 日 下午 11:06"   (Chinese)
      "7 Nov 2024 @ 23:06"              (other locales)
    For the Chinese format we return None (locale-dependent, hard to parse reliably).
    """
    if not ts:
        return None
    # English: "Nov 7, 2024 @ 11:06pm"
    m = re.search(
        r"(\w{3})\s+(\d{1,2}),\s+(\d{4})\s+@\s+(\d{1,2}):(\d{2})(am|pm)",
        ts, re.IGNORECASE
    )
    if m:
        month_abbr, day, year, hour, minute, ampm = m.groups()
        _MONTHS = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        month = _MONTHS.get(month_abbr.capitalize(), 0)
        if month:
            h = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
            try:
                return datetime(int(year), month, int(day), h, int(minute),
                                tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


# ── DataFrame conversion ──────────────────────────────────────────────────────

def comments_to_dataframe(
    comments: list[dict],
    fallback_timestamp: "Optional[datetime]" = None,
    weight: float = 3.0,
):
    """
    Convert raw comment dicts to a DataFrame compatible with the review schema.

    Columns produced:
        review_id, review_content, voted_up, timestamp,
        playtime_hours, votes_up, votes_funny, source, weight
    The `source` column is set to "event_comment" to distinguish from reviews.
    The `weight` column is used by compute_sentiment_stats to give these
    comments higher influence than regular reviews in the sentiment average.

    Args:
        comments:           Raw comment dicts from fetch_event_comments().
        fallback_timestamp: Used when a comment's published_at cannot be parsed
                            (Steam renders timestamps via JS; they are often
                            absent in static HTML).  Typically set to the patch
                            note's published_at so comments are treated as
                            post-update data.
        weight:             Influence multiplier relative to a regular review
                            (weight=1.0).  Defaults to EVENT_COMMENT_WEIGHT from
                            config (3.0), meaning 1 comment ≈ 3 reviews.
    """
    import pandas as pd

    _COLS = ["review_id", "review_content", "voted_up",
             "timestamp", "playtime_hours", "votes_up", "votes_funny", "source", "weight"]

    if not comments:
        return pd.DataFrame(columns=_COLS)

    if fallback_timestamp is not None:
        _fb = pd.Timestamp(fallback_timestamp)
        if _fb.tzinfo is None:
            _fb = _fb.tz_localize("UTC")
        else:
            _fb = _fb.tz_convert("UTC")
    else:
        _fb = pd.NaT

    rows = []
    for i, c in enumerate(comments):
        content = c.get("content", "").strip()
        if not content:
            continue  # skip empty comments

        # Prefer parsed timestamp; fall back to patch note date
        pub = c.get("published_at")
        if pub is not None:
            ts = pd.Timestamp(pub, tz="UTC")
        else:
            ts = _fb

        rows.append({
            "review_id":      c.get("comment_id") or f"ec_{i}_{hash(content) & 0xFFFFFF}",
            "review_content": content,
            "voted_up":       True,        # event comments have no thumbs-down
            "timestamp":      ts,
            "playtime_hours": 0.0,
            "votes_up":       c.get("upvotes", 0),
            "votes_funny":    0,
            "source":         "event_comment",
            "weight":         weight,
        })

    if not rows:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(rows)
    # Drop rows with no usable timestamp (both published_at and fallback are None)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df
