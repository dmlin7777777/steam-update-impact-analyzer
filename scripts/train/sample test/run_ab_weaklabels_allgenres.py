"""
Run weak-label generation and AB training across all genres and flag variants found under `features/`.

Behavior:
- Walk `features/` for parquet files
- For each parquet: determine genre (parent folder) and flag-variant (in filename)
- If weak-label columns missing, generate lightweight weak labels and save a new parquet with suffix `_with_weaklabels.parquet`
- Call existing `run_task` from `run_ab_weaklabels.py` to train risk and trend for that dataset
- Save outputs to `analysis_results/train/ab_weaklabels/<genre>/<flag>/`

This script is intentionally conservative: it will not overwrite an existing `_with_weaklabels.parquet` file unless `--overwrite` is passed.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import importlib.util
import numpy as np
import pandas as pd


ROOT = Path(r"c:/Users/12932/Desktop/nus/BAP")
FEATURES_ROOT = ROOT / "features"
OUT_ROOT = ROOT / "analysis_results/train/ab_weaklabels"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def detect_flag_variant(fname: str) -> str:
    n = fname.lower()
    if 'exclflag' in n or 'exclflagged' in n:
        return 'exclflagged'
    if 'inclflag' in n or 'inclflagged' in n:
        return 'inclflagged'
    if 'flag' in n and 'no' not in n:
        return 'flagged'
    return 'noflag'


# Note: weak-label generation is intentionally not performed here.
# This runner groups existing *_with_weaklabels.parquet files by flag variant
# and trains on the concatenated data for each group.


def main(parquet_root: Path, no_train: bool = False):
    """Programmatic runner that groups existing *_with_weaklabels.parquet files
    by flag variant and trains each group together.

    - parquet_root: Path to the `features/` directory
    - no_train: if True, does not run training and only returns a summary
    """
    # dynamic import of existing run_task to avoid circular behavior
    # prefer to load from same directory as this script (scripts/train/)
    run_ab_path = Path(__file__).parent / 'run_ab_weaklabels.py'
    # fallback: try one level up (scripts/), for safety
    if not run_ab_path.exists():
        run_ab_path = Path(__file__).parents[1] / 'run_ab_weaklabels.py'
    if not run_ab_path.exists():
        raise FileNotFoundError(f"Could not find run_ab_weaklabels.py near {__file__}; looked at {Path(__file__).parent} and {Path(__file__).parents[1]}")
    spec = importlib.util.spec_from_file_location('run_ab', str(run_ab_path))
    run_ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_ab)

    parquets = list(parquet_root.rglob('*.parquet'))
    if not parquets:
        print(f'No parquet files found under {parquet_root}')
        return {}

    # collect only augmented parquets (those that already have weak labels)
    aug_files = [p for p in parquets if '_with_weaklabels' in p.name.lower()]

    groups = {'inclflagged': [], 'exclflagged': []}
    for p in aug_files:
        n = p.name.lower()
        if 'inclflag' in n or 'inclflagged' in n:
            groups['inclflagged'].append(p)
        elif 'exclflag' in n or 'exclflagged' in n:
            groups['exclflagged'].append(p)

    summary = {}
    for flag, files in groups.items():
        outdir = OUT_ROOT / flag
        outdir.mkdir(parents=True, exist_ok=True)
        if not files:
            summary[flag] = {'num_files': 0, 'note': 'no_files_found'}
            continue

        print(f'Preparing group {flag} with {len(files)} files...')
        dfs = []
        read_errors = []
        for p in files:
            try:
                df = pd.read_parquet(p)
                dfs.append(df)
            except Exception as e:
                read_errors.append({str(p): str(e)})

        if not dfs:
            summary[flag] = {'num_files': len(files), 'read_errors': read_errors}
            continue

        # concat along rows and drop all-NaN columns
        combined = pd.concat(dfs, ignore_index=True)
        all_na = [c for c in combined.columns if combined[c].isna().all()]
        if all_na:
            print(f'Dropping all-NaN columns from combined df for {flag}: {all_na}')
            combined = combined.drop(columns=all_na)

        if no_train:
            summary[flag] = {'num_files': len(files), 'note': 'no_train', 'read_errors': read_errors}
            continue

        try:
            print(f'Running training for group {flag} (rows={len(combined)})...')
            run_ab.run_task(combined, 'risk_label_weak', outdir)
            run_ab.run_task(combined, 'trend_alert_weak', outdir)
            try:
                with open(outdir / 'metrics.json', 'r', encoding='utf-8') as f:
                    m = json.load(f)
            except Exception:
                m = None
            summary[flag] = {'num_files': len(files), 'metrics': m, 'read_errors': read_errors}
        except Exception as e:
            summary[flag] = {'error': str(e), 'read_errors': read_errors}

    # save a simple summary report
    with open(OUT_ROOT / 'all_genres_report.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': summary}, f, ensure_ascii=False, indent=2)

    print('Done. Summary written to', OUT_ROOT / 'all_genres_report.json')
    return summary


if __name__ == '__main__':
    features_dir = Path(r"c:\Users\12932\Desktop\nus\BAP\features")
    summary = main(features_dir, no_train=False)  # no_train=True 可仅检查而不跑训练
    print(summary)
