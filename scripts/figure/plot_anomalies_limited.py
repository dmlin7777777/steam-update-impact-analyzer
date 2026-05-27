import os
from typing import Optional
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


def sample_anomalies(df: pd.DataFrame, severity_col: str = 'anomaly_severity', is_anom_col: str = 'is_ensemble_anomaly', max_points: int = 200) -> pd.DataFrame:
    """Return a sampled subset of anomaly rows prioritizing high -> medium -> low.

    If fewer anomalies than max_points, returns all anomalies.
    """
    if is_anom_col not in df.columns:
        return pd.DataFrame()

    anoms = df[df[is_anom_col] == 1].copy()
    if anoms.empty:
        return anoms

    # If severity column exists, prioritize
    if severity_col in anoms.columns:
        # ensure categorical order
        order = {'high': 0, 'medium': 1, 'low': 2}
        anoms['_prio'] = anoms[severity_col].map(order).fillna(3).astype(int)
        anoms = anoms.sort_values(['_prio'], ascending=True)
    else:
        anoms = anoms.copy()

    if len(anoms) <= max_points:
        return anoms.drop(columns=['_prio'], errors='ignore')

    # sample balanced across severity levels if available
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

    # if still space or no severity info, sample the rest uniformly
    if remaining > 0:
        others = anoms.drop(pd.concat(sampled_parts).index if sampled_parts else [])
        if not others.empty:
            take = min(remaining, len(others))
            sampled_parts.append(others.sample(n=take, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    # if we overshot due to rounding, truncate
    if len(sampled) > max_points:
        sampled = sampled.sample(n=max_points, random_state=42)

    return sampled.drop(columns=['_prio'], errors='ignore')


def plot_limited_anomalies(df: pd.DataFrame, out_path: str, max_anoms: int = 200, time_col: str = 'review_date'):
    """Create and save the anomaly-limited visualization."""
    # Ensure time column
    df = df.copy()
    if time_col in df.columns:
        df['__ts'] = pd.to_datetime(df[time_col])
    else:
        # fallback to index order
        df['__ts'] = pd.RangeIndex(start=0, stop=len(df))

    # sample anomalies
    sampled_anoms = sample_anomalies(df, max_points=max_anoms)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Top: time-series sentiment with anomaly markers (limited)
    ax = axes[0]
    # plot background sentiment line (daily smoothing to match minimal granularity)
    # Use '1d' for daily resampling to avoid hourly resample when data is daily
    ts = df.set_index('__ts').resample('1d')['vader_compound'].mean().interpolate()
    ax.plot(ts.index, ts.values, color='C0', label='daily_mean_sentiment')

    if not sampled_anoms.empty:
        # plot anomalies; jitter x slightly if many points overlap
        # for daily-granularity data, jitter by up to +/- 0.25 days (~6 hours)
        jitter = (np.random.RandomState(42).rand(len(sampled_anoms)) - 0.5) * pd.Timedelta(days=0.25)
        xvals = pd.to_datetime(sampled_anoms['__ts']) + jitter
        # map severity to colors
        if 'anomaly_severity' in sampled_anoms.columns:
            palette = {'low': '#ffcc00', 'medium': '#ff6600', 'high': '#cc0033'}
            colors = sampled_anoms['anomaly_severity'].map(palette).fillna('#888888')
        else:
            colors = ['#cc0033'] * len(sampled_anoms)

        ax.scatter(xvals, sampled_anoms['vader_compound'], c=colors, s=40, edgecolor='k', alpha=0.9, label='anomalies')

    ax.set_title('Sentiment Time Series with Limited Anomaly Markers')
    ax.set_ylabel('Vader Compound')
    ax.legend()

    # Bottom: scatter sentiment vs comment_rate, anomalies highlighted
    ax2 = axes[1]
    # main cloud (downsample if too large)
    total_points = len(df)
    if total_points > 2000:
        bg = df.sample(n=2000, random_state=1)
    else:
        bg = df

    ax2.scatter(bg['comment_rate_24h'], bg['vader_compound'], c='lightgray', s=10, alpha=0.6, label='background')

    if not sampled_anoms.empty:
        ax2.scatter(sampled_anoms['comment_rate_24h'], sampled_anoms['vader_compound'],
                    c=sampled_anoms.get('anomaly_severity', ['#cc0033']*len(sampled_anoms)).map({'low': '#ffcc00', 'medium': '#ff6600', 'high': '#cc0033'}) if 'anomaly_severity' in sampled_anoms.columns else '#cc0033',
                    s=50, edgecolor='k', label='sampled_anomalies')

    ax2.set_xlabel('comment_rate_24h')
    ax2.set_ylabel('vader_compound')
    ax2.set_title('Sentiment vs Comment Rate (anomalies limited)')
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved anomaly visualization to {out_path}")


def plot_limited_anomalies_from_path(file_path: str, out_path: Optional[str] = None, max_anoms: int = 200, time_col: str = 'review_date') -> str:
    """Load a features file from disk and run the plotting function.

    Args:
        file_path: path to .xlsx or .csv enhanced features file
        out_path: optional path for the output PNG; if None will be derived from file_path
        max_anoms: maximum anomaly points to plot
        time_col: time column name in the file

    Returns:
        The output PNG path
    """
    if file_path.lower().endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    if out_path is None:
        out_path = os.path.splitext(file_path)[0] + '_anomalies_limited.png'

    plot_limited_anomalies(df, out_path, max_anoms=max_anoms, time_col=time_col)
    return out_path


def plot_per_app_from_path(file_path: str, out_dir: Optional[str] = None, max_anoms: int = 200, time_col: str = 'review_date') -> list:
    """Generate one anomaly visualization PNG per unique appid in the features file.

    Args:
        file_path: input .xlsx or .csv enhanced features file
        out_dir: directory to place per-app PNGs; if None, uses the input file directory
        max_anoms: maximum anomaly points to plot per app
        time_col: time column name

    Returns:
        List of generated output paths
    """
    if file_path.lower().endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    if 'appid' not in df.columns:
        raise ValueError("'appid' column not found in features file")

    if out_dir is None:
        out_dir = os.path.dirname(file_path) or '.'

    os.makedirs(out_dir, exist_ok=True)
    generated = []

    for appid, grp in df.groupby('appid'):
        safe_appid = str(appid).replace(os.sep, '_')
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}_{safe_appid}_anomalies_limited.png")
        try:
            plot_limited_anomalies(grp.reset_index(drop=True), out_path=out_path, max_anoms=max_anoms, time_col=time_col)
            generated.append(out_path)
        except Exception:
            # continue on errors per app to avoid blocking
            continue

    return generated


def main():
    """Main: select input file and output folder, then generate per-app plots.

    This uses tkinter dialogs when available; otherwise falls back to console prompts.
    """
    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title='Select enhanced features file', filetypes=[('Excel', '*.xlsx'), ('CSV', '*.csv')])
        if not file_path:
            return

        # choose output directory
        default_out_dir = os.path.dirname(file_path) or '.'
        out_dir = filedialog.askdirectory(title='Select output directory for per-app PNGs', initialdir=default_out_dir)
        if not out_dir:
            out_dir = default_out_dir

        max_anoms = simpledialog.askinteger('Max anomalies', 'Maximum anomaly points to plot per app', initialvalue=200, minvalue=1, maxvalue=5000)
        if max_anoms is None:
            max_anoms = 200

        try:
            generated = plot_per_app_from_path(file_path, out_dir=out_dir, max_anoms=int(max_anoms))
            if generated:
                msg = f'Generated {len(generated)} plots in:\n{out_dir}'
            else:
                msg = 'No plots were generated (no appid groups found or errors occurred).'
            messagebox.showinfo('Done', msg)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to generate per-app visualizations:\n{e}')
    else:
        # Fallback: console prompts
        print('Tkinter not available; falling back to console mode.')
        file_path = input('Path to enhanced features (.xlsx/.csv): ').strip()
        if not file_path:
            print('No file provided, exiting.')
            return
        out_dir = input('Output directory for per-app PNGs (enter to use input file folder): ').strip()
        if not out_dir:
            out_dir = os.path.dirname(file_path) or '.'
        try:
            max_anoms = int(input('Max anomalies to plot per app [200]: ') or 200)
        except Exception:
            max_anoms = 200
        generated = plot_per_app_from_path(file_path, out_dir=out_dir, max_anoms=max_anoms)
        if generated:
            print(f'Generated {len(generated)} plots in: {out_dir}')
        else:
            print('No plots were generated.')


if __name__ == '__main__':
    main()
