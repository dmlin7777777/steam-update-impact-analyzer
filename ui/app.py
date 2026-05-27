# ui/app.py
# Steam Update Impact Analyzer — Streamlit entry point.
#
# Run with:  streamlit run ui/app.py

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path when launched from ui/
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import ANALYSIS_POST_DAYS, ANALYSIS_PRE_DAYS, GAME_CATALOG
from core.scraper import SteamAPIError, fetch_app_details, fetch_patch_notes
from graph import app as pipeline_app
from ui.pages.dashboard import render_dashboard
from ui.pages.report import render_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steam Update Impact Analyzer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "selected_appid":  None,
    "game_info":       {},
    "patch_notes":     [],
    "analysis_result": None,
    "custom_games":    {},   # appid → {name, developer}
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_notes_cached(appid: str) -> list[dict]:
    """Fetch patch notes from Steam News API (cached 1 h)."""
    since = datetime.now(timezone.utc) - timedelta(days=180)
    return fetch_patch_notes(appid, since_dt=since, count=25)


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_app_info_cached(appid: str) -> dict:
    return fetch_app_details(appid)


def _select_game(appid: str) -> None:
    """Update session state when user picks a game."""
    if st.session_state["selected_appid"] == appid:
        return   # already selected, no re-fetch needed
    st.session_state["selected_appid"]  = appid
    st.session_state["analysis_result"] = None   # clear stale result

    all_games = {**GAME_CATALOG, **st.session_state["custom_games"]}
    st.session_state["game_info"] = all_games.get(appid, {})

    with st.spinner("Loading patch notes…"):
        try:
            st.session_state["patch_notes"] = _fetch_notes_cached(appid)
        except SteamAPIError:
            st.session_state["patch_notes"] = []


def _init_game_cache(appid: str, game_name: str) -> None:
    """Bulk-scrape up to 60 k reviews and write to local SQLite cache."""
    from core.bulk_scraper import bulk_scrape
    from core.local_cache import cache_date_range

    existing = cache_date_range(appid)
    if existing:
        st.info(
            f"Cache already has data for {game_name} "
            f"({existing[0].date()} → {existing[1].date()}). "
            f"Skipping initialization."
        )
        return

    counter = st.empty()
    progress = st.progress(0)
    MAX = 60_000

    def _cb(fetched: int, total: int) -> None:
        pct = min(total / MAX, 1.0)
        counter.caption(f"Fetched {total:,} / {MAX:,} reviews…")
        progress.progress(pct)

    with st.spinner(f"Initializing cache for {game_name}… (this may take a while)"):
        try:
            df = bulk_scrape(appid, max_reviews=MAX, progress_cb=_cb, store_in_cache=True)
            progress.empty()
            counter.empty()
            rng = cache_date_range(appid)
            st.success(
                f"✅ Cache initialized: {len(df):,} reviews stored "
                + (f"({rng[0].date()} → {rng[1].date()})" if rng else "")
            )
        except Exception as exc:
            progress.empty()
            counter.empty()
            st.warning(f"Cache initialization failed: {exc}. Analysis will use live API.")


def _run_pipeline(appid: str, update_date: datetime) -> None:
    """Execute the LangGraph pipeline with streaming progress display."""
    initial: dict = {
        "appid":            appid,
        "update_date":      update_date,
        "pre_days":         ANALYSIS_PRE_DAYS,
        "post_days":        ANALYSIS_POST_DAYS,
        "game_name":        st.session_state["game_info"].get("name", ""),
        "pre_reviews":      None,
        "post_reviews":     None,
        "patch_notes":      [],
        "event_comments":   None,
        "cleaned_reviews":  None,
        "flagged_reviews":  None,
        "analysis":         None,
        "llm_review_log":   [],
        "recommendations":  None,
        "current_step":     "start",
        "errors":           [],
    }

    _NODE_LABELS = {
        "scraper":        "🌐 Fetching reviews & patch notes",
        "cleaning":       "🧹 Cleaning reviews",
        "llm_review":     "🤖 LLM: reviewing flagged content",
        "analysis":       "📊 Running sentiment & topic analysis",
        "llm_sentiment":  "🤖 LLM: re-scoring ambiguous reviews",
        "recommendation": "✍️  LLM: generating report",
    }

    final_state = initial.copy()

    with st.status("Running analysis…", expanded=True) as status:
        try:
            for chunk in pipeline_app.stream(initial):
                node_name = next(iter(chunk))
                label     = _NODE_LABELS.get(node_name, node_name)
                st.write(f"✅ {label}")
                # Merge chunk into final state
                final_state.update(chunk.get(node_name, {}))

            errors = final_state.get("errors", [])
            if errors:
                # Surface non-fatal warnings
                for err in errors:
                    st.warning(err)

            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            st.session_state["analysis_result"] = final_state

        except Exception as exc:
            status.update(label=f"❌ Pipeline error: {exc}", state="error")
            st.error(str(exc))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎮 Games")
    st.caption("Select a game to view update history and run analysis.")

    all_games = {**GAME_CATALOG, **st.session_state["custom_games"]}
    selected  = st.session_state["selected_appid"]

    for appid, info in all_games.items():
        label = info["name"] if isinstance(info, dict) else info
        is_active = appid == selected
        btn_type  = "primary" if is_active else "secondary"
        if st.button(label, key=f"btn_{appid}", use_container_width=True, type=btn_type):
            _select_game(appid)
            st.rerun()

    st.divider()

    # ── Add custom game ───────────────────────────────────────────────────────
    with st.expander("➕ Add game by AppID"):
        custom_id = st.text_input("Steam AppID", placeholder="e.g. 292030", key="custom_id_input")
        init_cache = st.checkbox(
            "Initialize review cache (up to 60 k reviews, ~5–30 min)",
            value=False,
            key="init_cache_checkbox",
            help="Scrapes the most recent 60,000 reviews and stores them locally. "
                 "Required for historical analysis beyond the live API window (~30 days). "
                 "Skip if you only need recent updates.",
        )
        if st.button("Add", key="add_custom_game") and custom_id.strip():
            appid_str = custom_id.strip()
            if appid_str in all_games:
                st.info("Already in the list.")
            else:
                with st.spinner("Verifying…"):
                    try:
                        info = _fetch_app_info_cached(appid_str)
                        st.session_state["custom_games"][appid_str] = info
                        st.success(f"Added: {info['name']}")
                    except SteamAPIError as exc:
                        st.error(f"Could not fetch app info: {exc}")
                        st.stop()

                if init_cache:
                    _init_game_cache(appid_str, info.get("name", appid_str))

                st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

if st.session_state["selected_appid"] is None:
    # ── Welcome screen ────────────────────────────────────────────────────────
    st.markdown(
        """
        # 🎮 Steam Update Impact Analyzer
        ### Understand how game updates affect player sentiment

        **How it works:**
        1. Select a game from the sidebar
        2. Browse its recent update announcements
        3. Click **Analyze this update** — the pipeline fetches reviews before & after,
           runs NLP analysis, and generates a Claude-powered recommendation report
        """
    )
    st.info("👈  Pick a game from the sidebar to get started.")

else:
    appid     = st.session_state["selected_appid"]
    info      = st.session_state["game_info"]
    game_name = info.get("name", f"App {appid}") if isinstance(info, dict) else str(info)

    # ── Game header ───────────────────────────────────────────────────────────
    header_img = info.get("header_image", "") if isinstance(info, dict) else ""
    if header_img:
        h1, h2 = st.columns([1, 4])
        with h1:
            st.image(header_img, use_container_width=True)
        with h2:
            st.title(game_name)
            developer = info.get("developer", "") if isinstance(info, dict) else ""
            if developer:
                st.caption(f"Developer: **{developer}** · AppID: `{appid}`")
    else:
        st.title(f"🎮 {game_name}")
        st.caption(f"AppID: `{appid}`")

    tab_updates, tab_dashboard, tab_report = st.tabs([
        "📋 Update Announcements",
        "📊 Dashboard",
        "📄 Analysis Report",
    ])

    # ── Tab 1: Update Announcements ───────────────────────────────────────────
    with tab_updates:
        patch_notes = st.session_state.get("patch_notes", [])

        # Manual date picker (always available as fallback)
        with st.expander("🗓️  Run analysis for a custom date", expanded=not patch_notes):
            col_date, col_go = st.columns([3, 1])
            with col_date:
                custom_date = st.date_input(
                    "Update date",
                    value=datetime.now(timezone.utc).date() - timedelta(days=7),
                    key="custom_date",
                )
            with col_go:
                st.write("")  # vertical alignment spacer
                if st.button("▶ Analyze", key="run_custom", type="primary"):
                    update_dt = datetime(
                        custom_date.year, custom_date.month, custom_date.day,
                        tzinfo=timezone.utc,
                    )
                    _run_pipeline(appid, update_dt)
                    st.rerun()

        st.divider()

        if not patch_notes:
            st.info(
                "No patch notes found in the last 6 months for this game. "
                "Use the custom date picker above to run an analysis."
            )
        else:
            st.subheader(f"Recent Updates ({len(patch_notes)} found)")
            for note in patch_notes:
                pub_dt = note["published_at"] if isinstance(note, dict) else note.published_at
                title  = note["title"]        if isinstance(note, dict) else note.title
                url    = note["url"]          if isinstance(note, dict) else note.url
                body   = note["contents"]     if isinstance(note, dict) else note.contents

                with st.container(border=True):
                    row1, row2 = st.columns([5, 2])
                    with row1:
                        pub_str = pub_dt.strftime("%Y-%m-%d") if hasattr(pub_dt, "strftime") \
                                  else str(pub_dt)[:10]
                        st.markdown(f"**{title}**")
                        st.caption(f"📅 {pub_str}  ·  [View on Steam]({url})")
                        if body:
                            preview = body[:250].replace("\n", " ")
                            st.markdown(f"<small>{preview}…</small>", unsafe_allow_html=True)
                    with row2:
                        gid_key = note["gid"] if isinstance(note, dict) else note.gid
                        if st.button("🔍 Analyze", key=f"analyze_{gid_key}", type="primary"):
                            if isinstance(pub_dt, datetime):
                                update_dt = pub_dt
                            else:
                                update_dt = datetime(
                                    pub_dt.year, pub_dt.month, pub_dt.day,
                                    tzinfo=timezone.utc,
                                )
                            _run_pipeline(appid, update_dt)
                            st.rerun()

    # ── Tab 2: Dashboard ──────────────────────────────────────────────────────
    with tab_dashboard:
        result = st.session_state.get("analysis_result")
        if result:
            render_dashboard(result)
        else:
            st.info("Run an analysis from the **Update Announcements** tab to see the dashboard.")

    # ── Tab 3: Report ─────────────────────────────────────────────────────────
    with tab_report:
        result = st.session_state.get("analysis_result")
        if result:
            render_report(result)
        else:
            st.info("Run an analysis from the **Update Announcements** tab to generate a report.")
