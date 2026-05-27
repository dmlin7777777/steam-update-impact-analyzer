import os
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

sns.set(style='whitegrid')


def sample_anomalies(df: pd.DataFrame, is_anom_col: str = 'is_ensemble_anomaly', severity_col: str = 'anomaly_severity', max_points: int = 200) -> pd.DataFrame:
    """Sample anomaly rows prioritizing high->medium->low severity (if present).

    Returns a DataFrame of at most max_points anomaly rows. If there are fewer
    anomalies than max_points returns them all. If the is_anom column is missing
    returns an empty DataFrame.
    """
    if is_anom_col not in df.columns:
        return pd.DataFrame()

    anoms = df[df[is_anom_col] == 1].copy()
    if anoms.empty:
        return anoms

    if severity_col in anoms.columns:
        order = {'high': 0, 'medium': 1, 'low': 2}
        anoms['_prio'] = anoms[severity_col].map(order).fillna(3).astype(int)
        anoms = anoms.sort_values(['_prio'], ascending=True)

    if len(anoms) <= max_points:
        return anoms.drop(columns=['_prio'], errors='ignore')

    sampled_parts = []
    remaining = max_points
    if severity_col in anoms.columns:
        for sev in ['high', 'medium', 'low']:
            part = anoms[anoms[severity_col] == sev]
            if part.empty:
                continue
            take = int(np.ceil((len(part) / len(anoms)) * max_points))
            take = min(take, remaining)
            if take <= 0:
                continue
            sampled_parts.append(part.sample(n=take, random_state=42))
            remaining -= take
            if remaining <= 0:
                break

    if remaining > 0:
        others = anoms.drop(pd.concat(sampled_parts).index if sampled_parts else [])
        if not others.empty:
            take = min(remaining, len(others))
            sampled_parts.append(others.sample(n=take, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_points:
        sampled = sampled.sample(n=max_points, random_state=42)

    return sampled.drop(columns=['_prio'], errors='ignore')


def boxplot_by_apps(df: pd.DataFrame, apps: List[int], appid_col: str, y_col: str, anom_col: str,
                    out_path: str, max_anoms: int = 200, title: Optional[str] = None,
                    base_sent: Optional[str] = None, datetime_col: Optional[str] = None,
                    combined: bool = True):
    """Create a grouped boxplot for the given app ids, overlaying sampled anomalies.

    - df: full dataframe
    - apps: list of appids to include (order preserved)
    - appid_col: column name for app id
    - y_col: numeric column to plot (sentiment rolling mean or similar)
    - anom_col: boolean/int column indicating anomaly membership
    - out_path: where to save the PNG/PDF
    - max_anoms: maximum anomaly points to show across all apps (will sample per-app proportionally)
    """
    # Prepare subset
    df_sub = df[df[appid_col].isin(apps)].copy()
    if df_sub.empty:
        raise ValueError('No rows for specified app ids')

    # Map appid to display names if available in df (column 'game' or provided mapping elsewhere)
    if 'game' in df_sub.columns:
        names = df_sub.drop_duplicates(subset=[appid_col]).set_index(appid_col)['game'].to_dict()
    else:
        names = {a: str(a) for a in apps}

    games = [names.get(a, str(a)) for a in apps]

    # Background data for boxplot
    data = [df_sub[df_sub[appid_col] == a][y_col].dropna().values for a in apps]

    # If combined, create left time-series (blue) + right boxplot (green)
    if combined:
        fig, (ax_ts, ax) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.6, 1]})
        # Left: time series aggregated across selected apps
        if base_sent and datetime_col and datetime_col in df.columns:
            try:
                ts_df = df[df[appid_col].isin(apps)].copy()
                ts_df[datetime_col] = pd.to_datetime(ts_df[datetime_col], errors='coerce')
                ts = ts_df.set_index(datetime_col).resample('1d')[base_sent].mean().interpolate()
                ax_ts.plot(ts.index, ts.values, color='#1f77b4', label='daily_mean_sentiment')
                # sampled anomalies on ts (less prominent)
                sampled_anoms = sample_anomalies(df[df[appid_col].isin(apps)], is_anom_col=anom_col,
                                                 severity_col='anomaly_severity', max_points=max_anoms)
                if not sampled_anoms.empty:
                    # jitter in days small
                    jitter = (np.random.RandomState(42).rand(len(sampled_anoms)) - 0.5) * pd.Timedelta(days=0.15)
                    xvals = pd.to_datetime(sampled_anoms[datetime_col]) + jitter
                    ax_ts.scatter(xvals, sampled_anoms.get(base_sent, sampled_anoms[y_col]),
                                  c='#1f77b4', s=30, alpha=0.6, edgecolor='none', label='anomalies')
                ax_ts.set_title('Time Series (selected apps)')
                ax_ts.set_ylabel(base_sent)
                ax_ts.grid(axis='y', alpha=0.25)
            except Exception:
                ax_ts.text(0.5, 0.5, 'Could not compute time series', ha='center')
        else:
            ax_ts.text(0.5, 0.5, 'No time column or base sentiment found', ha='center')
            ax_ts.set_axis_off()
        # Right: boxplot
        ax = ax
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(data, positions=list(range(1, len(apps) + 1)), widths=0.6, patch_artist=True, showfliers=False)

    # Colors: right boxplot in green; create per-app green shades
    base_green = '#2ca02c'
    colors = {a: base_green for a in apps}

    for patch, a in zip(bp['boxes'], apps):
        patch.set_facecolor(colors[a])
        patch.set_alpha(0.25)
    for median in bp['medians']:
        median.set_linewidth(2)

    # Overlay jittered points: normal vs sampled anomalies
    rng = np.random.default_rng(42)

    # Use the same severity-prioritized sampling as plot_anomalies_limited:
    # globally sample up to max_anoms anomalies prioritizing high->medium->low,
    # then display the sampled anomalies on their respective app boxplots.
    sampled_anoms = sample_anomalies(df_sub, is_anom_col=anom_col, severity_col='anomaly_severity', max_points=max_anoms)

    for i, a in enumerate(apps, start=1):
        sub = df_sub[df_sub[appid_col] == a]
        # normal points
        if anom_col in sub.columns:
            if sub[anom_col].dtype == bool:
                normal = sub[~sub[anom_col]]
            else:
                normal = sub[sub[anom_col] == 0]
        else:
            normal = sub

        ax.scatter(rng.normal(i, 0.04, size=len(normal)), normal[y_col], s=12, alpha=0.12, color=colors[a], zorder=1)

    if not sampled_anoms.empty:
        # anomalies less prominent: use green shades, smaller size, lower alpha
        for i, a in enumerate(apps, start=1):
            ssub = sampled_anoms[sampled_anoms[appid_col] == a]
            if ssub.empty:
                continue
            # color anomalies same green (or map severity to slightly different greens if desired)
            s_color = colors[a]
            ax.scatter(rng.normal(i, 0.02, size=len(ssub)), ssub[y_col],
                       s=40, alpha=0.6, c=s_color, edgecolor='none', zorder=2)

    ax.set_xticks(list(range(1, len(apps) + 1)))
    ax.set_xticklabels(games)
    ax.set_ylabel(y_col)
    ax.set_title(title or f'Boxplot of {y_col} by App')
    ax.grid(axis='y', alpha=0.3)

    # Legend
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[a], markersize=10, alpha=0.6) for a in apps]
    labels = games
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, title='Apps')

    fig.tight_layout(rect=[0, 0, 0.84, 1])

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved boxplot to {out_path}.png and {out_path}.pdf')


def load_features_file(path: str) -> pd.DataFrame:
    """Load a features file (parquet / xlsx / csv) into a DataFrame."""
    path = str(path)
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    if path.lower().endswith('.xlsx') or path.lower().endswith('.xls'):
        return pd.read_excel(path)
    # fallback to csv
    return pd.read_csv(path)


def plot_from_path(file_path: str, apps: Optional[List[int]] = None, out_dir: Optional[str] = None,
                   appid_col: str = 'appid', y_col: Optional[str] = None, anom_col: str = 'is_ensemble_anomaly',
                   max_anoms: int = 200):
    """Load features and produce boxplots. If apps is None, use top 2 appids by count."""
    df = load_features_file(file_path)

    # guess app column
    if appid_col not in df.columns:
        alt = None
        for c in ['app_id', 'app']:
            if c in df.columns:
                alt = c; break
        if alt:
            appid_col = alt
        else:
            raise ValueError('No appid column found in file')

    # guess y_col
    # We'll prefer a derived deviation metric (z-score relative to rolling median/std)
    # because it highlights outliers better than a rolling mean. We compute it
    # from an available base sentiment column (prefer 'vader_compound').
    if y_col is None:
        base_sent = None
        for cand in ['vader_compound', 'sentiment_rolling_mean_72h', 'sentiment_rolling_mean_24h']:
            if cand in df.columns:
                base_sent = cand
                break
        if base_sent is None:
            # fallback: pick first numeric column besides appid
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != appid_col]
            if not numeric_cols:
                raise ValueError('Could not infer a numeric sentiment column for boxplot')
            base_sent = numeric_cols[0]

        # determine datetime column if present to sort within app groups
        datetime_col = None
        for dc in ['review_datetime', 'review_date', 'date', 'timestamp']:
            if dc in df.columns:
                datetime_col = dc
                break

        dev_col = 'sentiment_dev_zscore_72'
        if dev_col not in df.columns:
            df[dev_col] = np.nan
            # compute per-app rolling-median zscore
            if appid_col not in df.columns:
                # compute globally if no appid
                grp_iter = [(None, df.index)]
            else:
                grp_iter = [(appid, idx) for appid, idx in df.groupby(appid_col).groups.items()]

            for item in grp_iter:
                if item[0] is None:
                    idx = item[1]
                    sub = df.loc[idx]
                else:
                    idx = item[1]
                    sub = df.loc[idx]

                if datetime_col:
                    sub = sub.sort_values(datetime_col)

                series = pd.to_numeric(sub[base_sent], errors='coerce').fillna(method='ffill').fillna(method='bfill')
                n = len(series)
                if n == 0:
                    continue
                window = min(72, max(3, n))
                med = series.rolling(window=window, center=True, min_periods=1).median()
                std = series.rolling(window=window, center=True, min_periods=1).std(ddof=0)
                # fall back to global std if local std is zero/NaN
                global_std = series.std(ddof=0)
                std = std.fillna(global_std).replace(0, global_std if global_std != 0 else 1.0)
                dev = (series - med) / std
                # assign back preserving original index order
                df.loc[sub.index, dev_col] = dev.values

        y_col = dev_col

    # guess anomaly column
    if anom_col not in df.columns:
        for cand in ['is_ensemble_anomaly', 'ml_anomaly', 'is_statistical_anomaly']:
            if cand in df.columns:
                anom_col = cand; break

    if apps is None:
        # pick top 2 appids by row count
        top = df[appid_col].value_counts().nlargest(2).index.tolist()
        apps = top

    if out_dir is None:
        out_dir = os.path.dirname(file_path) or '.'

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(out_dir, f"{base}_boxplot_{'_'.join(map(str, apps))}")

    boxplot_by_apps(df, apps, appid_col=appid_col, y_col=y_col, anom_col=anom_col, out_path=out_path, max_anoms=max_anoms)
    return out_path + '.png'


def main():
    # 非交互 main：把要处理的文件地址写在这里，直接生成箱型图
    # 修改下面的 file_path 为你想要处理的文件（支持 .parquet/.xlsx/.csv）
    file_path = r"c:\Users\12932\Desktop\nus\BAP\features\leisure\gpu_optimized_features_leisure_inclflagged_enhanced_features.xlsx"
    out_dir = os.path.dirname(file_path) or '.'
    # 如果你想指定特定 appids，请把下面的 apps 改为列表，例如 [413150, 945360]
    apps = None  # None 表示使用 top 2 appids
    max_anoms = 200

    print(f"Running boxplot generator on: {file_path}\nOutput dir: {out_dir}\nApps: {apps or 'top 2'}\nMax anomalies: {max_anoms}")
    try:
        out_png = plot_from_path(file_path, apps=apps, out_dir=out_dir, max_anoms=max_anoms)
        print('Saved:', out_png)
    except Exception as e:
        print('Failed to create boxplot:', e)


if __name__ == '__main__':
    main()
