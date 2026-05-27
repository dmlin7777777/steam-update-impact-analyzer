# core/local_cache.py
# SQLite-backed local review cache for pre-scraped Steam data.
#
# Why this exists
# ---------------
# Steam's review API returns data newest-first with no date-range parameter.
# For high-traffic games (CS2, Dota 2) a historical window > ~30 days ago
# cannot be reached within a small live-scraping budget.  The solution is an
# offline bulk-scrape → local SQLite store → fast range query pattern:
#
#   1. Run scripts/import_cache.py once to load your existing CSVs/Excel files.
#   2. Future bulk scrapes can be imported the same way.
#   3. core/scraper.fetch_reviews() checks the cache first; live API is the
#      fallback for windows not yet in the cache.
#
# Schema
# ------
# Table: reviews
#   appid          TEXT   Steam App ID
#   review_id      TEXT   recommendationid (or surrogate key for legacy data)
#   review_content TEXT   raw review text
#   voted_up       INT    1 = positive, 0 = negative
#   timestamp      TEXT   ISO-8601 UTC (sortable: "YYYY-MM-DD HH:MM:SS")
#   playtime_hours REAL   hours played at review time
#   votes_up       INT
#   votes_funny    INT
#
# Index: (appid, timestamp) for fast range queries.

from __future__ import annotations

import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import REVIEW_CACHE_DB

_EMPTY_COLUMNS = [
    "review_id", "review_content", "voted_up",
    "timestamp", "playtime_hours", "votes_up", "votes_funny",
]

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS reviews (
    appid          TEXT NOT NULL,
    review_id      TEXT NOT NULL,
    review_content TEXT NOT NULL,
    voted_up       INT  NOT NULL,
    timestamp      TEXT NOT NULL,
    playtime_hours REAL NOT NULL DEFAULT 0,
    votes_up       INT  NOT NULL DEFAULT 0,
    votes_funny    INT  NOT NULL DEFAULT 0,
    PRIMARY KEY (appid, review_id)
);
CREATE INDEX IF NOT EXISTS idx_appid_ts ON reviews (appid, timestamp);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    db_path = Path(REVIEW_CACHE_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


# ── Initialisation ────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the reviews table and index if they do not yet exist."""
    with _connect() as conn:
        conn.executescript(_CREATE_SQL)


# ── Import helpers ────────────────────────────────────────────────────────────

def _normalise_legacy_df(df: pd.DataFrame, appid: str) -> pd.DataFrame:
    """
    Convert an old-style scraper DataFrame (scrapper_optimized.py schema) to
    the canonical review schema expected by this cache.

    Old schema key columns:
        SteamID, played_hours (minutes), playtime_at_review (minutes),
        review_content, voted_up (1/0), review_date (YYYY-MM-DD str),
        votes_up, votes_funny
    """
    out = pd.DataFrame()

    # review_id: use SteamID + review_date as surrogate (deterministic)
    sid  = df["SteamID"].astype(str)
    rdate = df["review_date"].astype(str)
    out["review_id"] = (sid + "_" + rdate).apply(
        lambda s: hashlib.md5(s.encode()).hexdigest()[:16]
    )

    out["review_content"] = df["review_content"].fillna("").astype(str)

    # voted_up: stored as 1/0 int in old schema
    out["voted_up"] = df["voted_up"].fillna(0).astype(int)

    # timestamp: review_date is "YYYY-MM-DD", interpret as UTC midnight
    out["timestamp"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)

    # playtime_hours: prefer playtime_at_review (at review time), fallback played_hours
    # Both are stored in MINUTES in Steam API
    if "playtime_at_review" in df.columns:
        pa = pd.to_numeric(df["playtime_at_review"], errors="coerce").fillna(0)
    else:
        pa = pd.Series(0, index=df.index, dtype=float)

    ph = pd.to_numeric(df.get("played_hours", 0), errors="coerce").fillna(0)
    minutes = pa.where(pa > 0, ph)
    out["playtime_hours"] = (minutes / 60).round(1)

    out["votes_up"]    = pd.to_numeric(df.get("votes_up",    0), errors="coerce").fillna(0).astype(int)
    out["votes_funny"] = pd.to_numeric(df.get("votes_funny", 0), errors="coerce").fillna(0).astype(int)
    out["appid"]       = str(appid)

    # Drop rows where timestamp couldn't be parsed or content is empty
    out = out.dropna(subset=["timestamp"])
    out = out[out["review_content"].str.strip() != ""]
    return out


def _normalise_new_df(df: pd.DataFrame, appid: str) -> pd.DataFrame:
    """
    Convert a new-style scraper DataFrame (core/scraper.py schema) to cache
    format.  New schema already uses: review_id, review_content, voted_up,
    timestamp, playtime_hours, votes_up, votes_funny.
    """
    out = df[["review_id", "review_content", "voted_up",
              "timestamp", "playtime_hours", "votes_up", "votes_funny"]].copy()
    out["appid"]    = str(appid)
    out["voted_up"] = out["voted_up"].astype(int)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    return out


def import_dataframe(df: pd.DataFrame, appid: str, legacy: bool = True) -> int:
    """
    Insert rows from *df* into the cache for *appid*.
    Duplicate (appid, review_id) pairs are silently ignored (INSERT OR IGNORE).

    Args:
        df:     Source DataFrame (legacy or new schema).
        appid:  Steam App ID string.
        legacy: True if df uses the old scrapper_optimized.py column names.

    Returns:
        Number of new rows inserted.
    """
    init_db()
    normalised = _normalise_legacy_df(df, appid) if legacy else _normalise_new_df(df, appid)

    if normalised.empty:
        return 0

    # Format timestamp as sortable ISO string
    normalised["timestamp"] = normalised["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Build list of tuples for executemany — avoids SQLite variable-count limits
    cols = ["appid", "review_id", "review_content", "voted_up",
            "timestamp", "playtime_hours", "votes_up", "votes_funny"]
    rows = [tuple(r) for r in normalised[cols].itertuples(index=False, name=None)]

    sql = """
        INSERT OR IGNORE INTO reviews
            (appid, review_id, review_content, voted_up,
             timestamp, playtime_hours, votes_up, votes_funny)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    _CHUNK = 2000  # rows per executemany call
    inserted = 0
    with _connect() as conn:
        for i in range(0, len(rows), _CHUNK):
            batch = rows[i : i + _CHUNK]
            cur = conn.executemany(sql, batch)
            inserted += cur.rowcount

    return inserted


def import_file(
    path: str | Path,
    appid: str,
    legacy: bool = True,
    sheet_name: int | str = 0,
) -> int:
    """
    Load a CSV or Excel file and import it into the cache.

    Args:
        path:       Path to .csv or .xlsx file.
        appid:      Steam App ID string.  Pass '' to auto-detect from an 'appid'
                    column in the file.
        legacy:     True for old scrapper_optimized.py schema.
        sheet_name: For Excel files with multiple sheets.

    Returns:
        Number of new rows inserted.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path, low_memory=False)

    # Auto-detect appid from column if not supplied
    if not appid and "appid" in df.columns:
        unique_ids = df["appid"].dropna().unique()
        total = 0
        for aid in unique_ids:
            subset = df[df["appid"] == aid]
            total += import_dataframe(subset, str(int(aid)), legacy=legacy)
        return total

    return import_dataframe(df, appid, legacy=legacy)


# ── Query ─────────────────────────────────────────────────────────────────────

def query_reviews(
    appid: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Return cached reviews for *appid* in [start_dt, end_dt] (both inclusive).

    Returns an empty DataFrame (correct columns) if no matching rows exist.
    """
    init_db()

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str   = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT review_id, review_content, voted_up, timestamp,
                   playtime_hours, votes_up, votes_funny
            FROM   reviews
            WHERE  appid = ?
              AND  timestamp >= ?
              AND  timestamp <= ?
            ORDER BY timestamp DESC
            """,
            (str(appid), start_str, end_str),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    df = pd.DataFrame(rows, columns=_EMPTY_COLUMNS)
    df["timestamp"]     = pd.to_datetime(df["timestamp"], utc=True)
    df["voted_up"]      = df["voted_up"].astype(bool)
    df["playtime_hours"] = df["playtime_hours"].astype(float)
    df["votes_up"]      = df["votes_up"].astype(int)
    df["votes_funny"]   = df["votes_funny"].astype(int)
    return df


def cache_date_range(appid: str) -> Optional[tuple[datetime, datetime]]:
    """
    Return (min_date, max_date) of cached reviews for *appid*, or None if empty.
    """
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM reviews WHERE appid=?",
            (str(appid),),
        ).fetchone()

    if not row or row[0] is None:
        return None

    fmt = "%Y-%m-%d %H:%M:%S"
    return (
        datetime.strptime(row[0], fmt).replace(tzinfo=timezone.utc),
        datetime.strptime(row[1], fmt).replace(tzinfo=timezone.utc),
    )


def cache_row_count(appid: str) -> int:
    """Return number of cached reviews for *appid*."""
    init_db()
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE appid=?", (str(appid),)
        ).fetchone()[0]
