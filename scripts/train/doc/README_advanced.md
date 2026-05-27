# Advanced training script

This document explains the `advanced_train.py` script in this folder.

Purpose
- Train a stronger model (LightGBM preferred) using numeric features with time-based cross validation and early stopping. Exports feature importance and a final model.

Usage
- Example:
```
python scripts/train/advanced_train.py --input path/to/features.parquet --target sentiment_label --time_col review_date --output_dir analysis_results/train/advanced
```

Notes
- LightGBM is used if installed; otherwise the script falls back to sklearn's `HistGradientBoostingClassifier`.
- Requires numeric features; if your features file lacks numeric columns, run the baseline with `--use_text` or generate numeric features first.
- Saves: model joblib, `metrics_advanced_<target>.json`, and `feature_importances_<target>.csv`.
