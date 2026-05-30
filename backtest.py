# backtest.py
# Lean backtesting script -- no LLM calls, pure VADER + risk signals.
# Runs each test case and compares actual outcome against expected.
#
# Usage:
#   python backtest.py              # run all cases
#   python backtest.py --case 0    # run single case by index

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.scraper import fetch_reviews
from core.features import extract_features
from core.analysis import compute_sentiment_stats, compute_risk_signals
from core.local_cache import cache_date_range

_SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

SEP  = "-" * 68
SEP2 = "=" * 68


def _overall_risk(signals) -> str:
    if not signals:
        return "LOW"
    return max(signals, key=lambda s: _SEVERITY_ORDER.get(s.severity, 0)).severity


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class BacktestCase:
    name:           str
    appid:          str
    update_date:    datetime
    expected_risk:  list      # acceptable risk levels, e.g. ["HIGH", "CRITICAL"]
    expected_delta: str       # "+" positive | "-" negative | "?" either
    rationale:      str
    pre_days:       int = 7
    post_days:      int = 7
    max_reviews:    int = 5_000


CASES = [
    BacktestCase(
        name="CS2 -- Knife Skin Trade-up Update",
        appid="730",
        update_date=datetime(2025, 10, 22, tzinfo=timezone.utc),
        expected_risk=["HIGH", "CRITICAL"],
        expected_delta="-",
        rationale="Knife skins added to trade-up -> secondary market crash, player anger",
        max_reviews=5_000,
    ),
    BacktestCase(
        name="Cyberpunk 2077 -- 2.0 Patch + Phantom Liberty",
        appid="1091500",
        update_date=datetime(2023, 9, 21, tzinfo=timezone.utc),
        expected_risk=["LOW", "MEDIUM"],
        expected_delta="+",
        rationale="Full skill-tree rework, AI overhaul -- community redemption moment",
        max_reviews=150_000,
    ),
    BacktestCase(
        name="Stardew Valley -- 1.6 Update",
        appid="413150",
        update_date=datetime(2024, 3, 19, tzinfo=timezone.utc),
        expected_risk=["LOW"],
        expected_delta="+",
        rationale="Free content update by solo dev -- universally celebrated",
        max_reviews=80_000,
    ),
    BacktestCase(
        name="Elden Ring -- Shadow of the Erdtree DLC",
        appid="1245620",
        update_date=datetime(2024, 6, 21, tzinfo=timezone.utc),
        expected_risk=["MEDIUM", "HIGH"],
        expected_delta="-",
        rationale="First-week difficulty controversy before community settled",
        max_reviews=120_000,
    ),
]


# ---------------------------------------------------------------------------
# Core analysis (no LLM)
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    case:          BacktestCase
    pre_n:         int
    post_n:        int
    pre_neg_pct:   float
    post_neg_pct:  float
    neg_delta:     float       # post neg% - pre neg% (positive = worsening)
    risk:          str
    signals:       list
    data_ok:       bool
    pass_risk:     Optional[bool] = None
    pass_delta:    Optional[bool] = None

    @property
    def passed(self) -> Optional[bool]:
        if not self.data_ok:
            return None
        return self.pass_risk and self.pass_delta


def run_case(case: BacktestCase) -> BacktestResult:
    pre_start = case.update_date - timedelta(days=case.pre_days)
    pre_end   = case.update_date - timedelta(seconds=1)
    post_end  = case.update_date + timedelta(days=case.post_days)

    pre_df  = fetch_reviews(case.appid, pre_start, pre_end,
                            max_reviews=case.max_reviews)
    post_df = fetch_reviews(case.appid, case.update_date, post_end,
                            max_reviews=case.max_reviews)

    data_ok = len(pre_df) >= 10 and len(post_df) >= 10

    if not pre_df.empty:
        pre_df  = extract_features(pre_df)
    if not post_df.empty:
        post_df = extract_features(post_df)

    pre_stats  = compute_sentiment_stats(pre_df)
    post_stats = compute_sentiment_stats(post_df)
    neg_delta  = round(post_stats.negative_pct - pre_stats.negative_pct, 4)

    # Build ReviewAnalysisResult for compute_risk_signals
    from agents.state import ReviewAnalysisResult
    review_result = ReviewAnalysisResult(sentiment=post_stats)
    signals    = compute_risk_signals(pre_stats, review_result, event_analysis=None)
    risk       = _overall_risk(signals)

    pass_risk  = (risk in case.expected_risk) if data_ok else None
    pass_delta = (
        (neg_delta > 0 if case.expected_delta == "-" else
         neg_delta < 0 if case.expected_delta == "+" else True)
        if data_ok else None
    )

    return BacktestResult(
        case=case,
        pre_n=len(pre_df), post_n=len(post_df),
        pre_neg_pct=pre_stats.negative_pct,
        post_neg_pct=post_stats.negative_pct,
        neg_delta=neg_delta, risk=risk, signals=signals,
        data_ok=data_ok,
        pass_risk=pass_risk,
        pass_delta=pass_delta,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _chk(val: Optional[bool]) -> str:
    if val is None:
        return "n/a"
    return "PASS" if val else "FAIL"


def print_result(r: BacktestResult) -> None:
    c = r.case
    print(SEP)
    print(f"  {c.name}")
    print(f"  AppID {c.appid}  |  {c.update_date.date()}")
    print(f"  {c.rationale}")
    print(SEP)

    if not r.data_ok:
        print(f"  [NO DATA]  pre={r.pre_n} reviews, post={r.post_n} reviews")
        rng = cache_date_range(c.appid)
        if rng:
            print(f"  Cache exists ({rng[0].date()} to {rng[1].date()}) but window not covered")
        else:
            print(f"  No cache for appid {c.appid} -- run bulk scrape first")
        return

    arrow = "^" if r.neg_delta > 0 else "v"
    print(f"  Pre-update    n={r.pre_n:>5,}   neg%={r.pre_neg_pct:.1%}")
    print(f"  Post-update   n={r.post_n:>5,}   neg%={r.post_neg_pct:.1%}")
    print(f"  Neg delta                {r.neg_delta:+.1%}  {arrow}")
    print(f"  Risk level    {r.risk}")
    if r.signals:
        for sig in r.signals:
            print(f"    [{sig.severity}] {sig.rule}: {sig.description}")

    exp_risk  = " or ".join(c.expected_risk)
    exp_delta = "negative" if c.expected_delta == "-" else \
                "positive" if c.expected_delta == "+" else "any"

    print()
    print(f"  Expected risk : {exp_risk:<20}  -> {_chk(r.pass_risk)}")
    print(f"  Expected delta: {exp_delta:<20}  -> {_chk(r.pass_delta)}")
    print(f"  Overall       : {'*** PASS ***' if r.passed else '!!! FAIL !!!'}")


def print_summary(results: list) -> None:
    runnable = [r for r in results if r.data_ok]
    passed   = [r for r in runnable if r.passed]
    skipped  = [r for r in results if not r.data_ok]

    print()
    print(SEP2)
    print("  BACKTEST SUMMARY")
    print(SEP2)
    print(f"  Total cases : {len(results)}")
    print(f"  Ran         : {len(runnable)}")
    print(f"  Passed      : {len(passed)} / {len(runnable)}")
    print(f"  Skipped     : {len(skipped)}  (no data)")

    if runnable:
        print()
        print(f"  {'Case':<45} {'Risk':<8}  {'Delta':>7}  Result")
        print(f"  {'-'*45} {'-'*8}  {'-'*7}  {'-'*6}")
        for r in results:
            if r.data_ok:
                res = "PASS" if r.passed else "FAIL"
                print(f"  {r.case.name:<45} {r.risk:<8}  {r.delta:>+7.4f}  {res}")
            else:
                print(f"  {r.case.name:<45} {'---':<8}  {'---':>7}  NO DATA")

    if skipped:
        print()
        print("  Games needing cache initialisation:")
        for r in skipped:
            rng = cache_date_range(r.case.appid)
            if rng:
                note = f"cache {rng[0].date()} to {rng[1].date()}, window outside range"
            else:
                note = f"no cache for appid {r.case.appid}"
            print(f"    * {r.case.name}  ({note})")
    print(SEP2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=None,
                        help="Run a single case by index (0-based)")
    args = parser.parse_args()
    cases = [CASES[args.case]] if args.case is not None else CASES

    print(SEP2)
    print("  STEAM UPDATE IMPACT ANALYSER -- BACKTEST")
    print(f"  {len(cases)} case(s)  |  no LLM calls  |  VADER + risk signals only")
    print(SEP2)

    results = []
    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case.name} ...", flush=True)
        t0 = time.time()
        r  = run_case(case)
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        results.append(r)
        print_result(r)

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
