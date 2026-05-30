from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Directories ───────────────────────────────────────────────────────────────
CACHE_DIR      = BASE_DIR / ".cache"               # API response cache
OUTPUT_DIR     = BASE_DIR / "output"               # analysis results, reports
REVIEW_CACHE_DB = BASE_DIR / "data" / "reviews.db" # local SQLite review cache

# ── Steam API endpoints ───────────────────────────────────────────────────────
STEAM_REVIEWS_URL     = "https://store.steampowered.com/appreviews/{appid}"
STEAM_NEWS_URL        = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
STEAM_APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"

# ── Curated game catalogue (appid → display info) ─────────────────────────────
# Covers multiple genres so the system is demonstrably general.
GAME_CATALOG: dict[str, dict] = {
    "730":     {"name": "Counter-Strike 2",  "developer": "Valve"},
    "570":     {"name": "Dota 2",            "developer": "Valve"},
    "1245620": {"name": "Elden Ring",         "developer": "FromSoftware"},
    "1091500": {"name": "Cyberpunk 2077",     "developer": "CD Projekt Red"},
    "1086940": {"name": "Baldur\'s Gate 3",   "developer": "Larian Studios"},
    "413150":  {"name": "Stardew Valley",     "developer": "ConcernedApe"},
    "367520":  {"name": "Hollow Knight",      "developer": "Team Cherry"},
    "1593500": {"name": "God of War",         "developer": "Santa Monica Studio"},
}

# ── Scraper settings ──────────────────────────────────────────────────────────
REVIEWS_PER_PAGE   = 100     # Steam API max per request
MAX_REVIEWS_TOTAL  = 500     # cap per window (pre or post)
SCRAPER_DELAY_SEC  = 0.5     # polite delay between paginated requests
SCRAPER_RETRIES    = 3

# ── Analysis window ───────────────────────────────────────────────────────────
ANALYSIS_PRE_DAYS  = 7       # days before update date  → baseline
ANALYSIS_POST_DAYS = 7       # days after  update date  → impact window

# ── NLP thresholds ────────────────────────────────────────────────────────────
# VADER compound score in (-VADER_GRAY_LO, VADER_GRAY_HI) → send to LLM
VADER_GRAY_LO = 0.2          # |compound| below this is ambiguous
LLM_REVIEW_THRESHOLD = 0.05  # flagged-review ratio that triggers LLM cleaning pass

# Topic labels used for Claude-based classification (genre-agnostic)
TOPIC_LABELS = [
    "performance",    # FPS drops, lag, crashes, load times
    "gameplay",       # mechanics, balance, controls, difficulty
    "content",        # new maps / missions / story / updates
    "bugs",           # glitches, soft-locks, reproducible errors
    "monetization",   # DLC pricing, microtransactions, battle pass
    "graphics",       # visuals, art style, resolution, RTX
    "audio",          # music, SFX, voice acting
    "community",      # multiplayer, matchmaking, toxicity
    "other",          # doesn't fit above
]

# ── Event comment blending ────────────────────────────────────────────────────
# Comments posted directly under an update announcement are the primary signal.
# General reviews from the nearby window are supplementary context.
#
# Blending strategy (fixed ratio, not per-row weight):
#   post_sentiment = EVENT_BLEND_RATIO  × event_comment_sentiment
#                  + (1 - EVENT_BLEND_RATIO) × review_sentiment
#
# Using a fixed ratio means event comments ALWAYS dominate regardless of raw
# counts — e.g. 500 event comments vs 5000 reviews still gives event comments
# 70% of the influence.  When no event comments are available, falls back to
# 100% reviews.
EVENT_COMMENT_WEIGHT = 3.0   # kept for DataFrame column; no longer used in stats
EVENT_BLEND_RATIO    = 0.70  # event comments → 70%, reviews → 30%

# ── Risk scoring rules ────────────────────────────────────────────────────────
ALERT_THRESHOLDS = {
    "sentiment_drop":         0.15,   # post−pre compound score drop
    "negative_surge_pct":     0.60,   # fraction of negative reviews
    "review_rate_multiplier": 3.0,    # post-update volume vs baseline
    "z_score_anomaly":        2.0,    # rolling z-score to flag as anomaly
}

# ── LLM models ────────────────────────────────────────────────────────────────
LLM_MODEL       = "claude-sonnet-4-6"           # deep analysis, recommendations
LLM_MODEL_LIGHT = "claude-haiku-4-5-20251001"   # cleaning review, topic tagging
