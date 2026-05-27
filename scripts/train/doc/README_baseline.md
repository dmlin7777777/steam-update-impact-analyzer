# Baseline training script

This document explains the `baseline_train.py` script in this folder.

Purpose
- Provide a simple, reproducible baseline using TF-IDF + LogisticRegression (if text used) or a numeric pipeline (median impute + scaler + LogisticRegression).

Usage
- Run from repository root (example):
```
python scripts/train/baseline_train.py --input path/to/features.parquet --target sentiment_label --time_col review_date --output_dir analysis_results/train/baseline
```

Notes
- Uses TimeSeriesSplit (default 3 splits) to estimate out-of-time performance.
- If `--use_text` is passed and `review_content_clean` exists, TF-IDF is used.
- Saves: model joblib and `metrics_baseline_<target>.json` to `output_dir`.
