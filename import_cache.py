#!/usr/bin/env python
# import_cache.py
# One-shot script to populate the local SQLite review cache from pre-scraped
# CSV / Excel files produced by scripts/scrapper/scrapper_optimized.py.
#
# Usage:
#   python scripts/import_cache.py                        # import defaults below
#   python scripts/import_cache.py path/to/file.csv 730  # custom file + appid
#
# After running, core/scraper.fetch_reviews() will serve cached data for any
# date window that falls within the imported range, with no network requests.

from __future__ import annotations

import sys
import io
from pathlib import Path

# Ensure project root is on the path when run from anywhere
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from core.local_cache import import_file, cache_date_range, cache_row_count, init_db

# ── Default import manifest ───────────────────────────────────────────────────
# Edit this list to match your local file paths.
# Each entry: (file_path, appid, legacy)
#   legacy=True  → old scrapper_optimized.py schema (SteamID, played_hours …)
#   legacy=False → new core/scraper.py schema (review_id, timestamp …)

_DEFAULT_IMPORTS: list[tuple[str, str, bool]] = [
    # Combined FPS dataset — covers CS2 (730) and 1938090 from Dec 2023 to Oct 2025
    ("data_nolabel/combined/combined_fps_reviews.xlsx", "", True),

    # Test data — CS2 reviews Oct 19-26 2025 (covers the knife trade-up update window)
    ("test data/reviews/cleaned_combined_fps_reviews.xlsx", "730", True),

    # Recent CS2 scrape — Jan to Apr 2026
    ("multi_games_data/730_reviews.csv", "730", True),
]


def _fmt_range(appid: str) -> str:
    r = cache_date_range(appid)
    if r is None:
        return "no data"
    return f"{r[0].date()} → {r[1].date()}"


def main(argv: list[str]) -> None:
    init_db()

    if len(argv) >= 3:
        # Manual invocation: import_cache.py <file> <appid>
        path  = Path(argv[1])
        appid = argv[2]
        legacy = True
        imports = [(str(path), appid, legacy)]
    else:
        imports = _DEFAULT_IMPORTS

    for file_path, appid, legacy in imports:
        p = _root / file_path
        if not p.exists():
            print(f"  ⚠  Not found, skipping: {p}")
            continue

        label = appid if appid else "(auto-detect from appid column)"
        print(f"\n→ Importing {p.name}  appid={label}  legacy={legacy}")
        try:
            inserted = import_file(p, appid, legacy=legacy)
            print(f"  ✓ Inserted {inserted:,} new rows")
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")

    # Summary
    print("\n=== Cache summary ===")
    from config import REVIEW_CACHE_DB
    print(f"DB path : {REVIEW_CACHE_DB}")

    import sqlite3
    with sqlite3.connect(REVIEW_CACHE_DB) as conn:
        rows = conn.execute(
            "SELECT appid, COUNT(*), MIN(timestamp), MAX(timestamp) "
            "FROM reviews GROUP BY appid ORDER BY appid"
        ).fetchall()

    if not rows:
        print("  (empty)")
    else:
        for appid, cnt, lo, hi in rows:
            print(f"  appid={appid:>8}  {cnt:>8,} rows  {lo[:10]} → {hi[:10]}")


if __name__ == "__main__":
    main(sys.argv)
