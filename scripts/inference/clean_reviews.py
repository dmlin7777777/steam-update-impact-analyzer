import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import html

def clean_text(s: str) -> str:
    if s is None:
        return ""
    # ensure string
    s = str(s)
    # unescape HTML entities
    s = html.unescape(s)
    # remove URLs
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    # remove html tags
    s = re.sub(r'<[^>]+>', ' ', s)
    # remove control characters
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    # lowercase
    s = s.lower()
    return s


def parse_time_column(value):
    """Try to parse different time formats. If year missing, append current year."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if s == '':
        return pd.NaT
    # remove @ and commas commonly present
    s = s.replace('@', '')
    s = s.replace(',', '')
    # append current year if no 4-digit year present
    if not re.search(r'\b\d{4}\b', s):
        s = f"{s} {datetime.now().year}"
    try:
        return pd.to_datetime(s, infer_datetime_format=True, dayfirst=False, errors='coerce')
    except Exception:
        # fallback to dateutil
        try:
            from dateutil import parser
            return parser.parse(s, fuzzy=True)
        except Exception:
            return pd.NaT


def detect_columns(df: pd.DataFrame):
    # find review column
    review_col = None
    time_col = None
    for c in df.columns:
        lc = c.lower()
        if 'review' in lc and review_col is None:
            review_col = c
        if 'time' in lc or 'date' in lc or 'timestamp' in lc:
            time_col = c
    # fallback to common names
    if review_col is None:
        for name in ['review_content', 'review', 'content', 'text']:
            if name in df.columns:
                review_col = name
                break
    if time_col is None:
        for name in ['time', 'date', 'timestamp']: 
            if name in df.columns:
                time_col = name
                break
    return review_col, time_col


def clean_file(input_path: str, output_path: str = None, save_parquet: bool = True):
    """Load CSV, clean review text and convert time, save cleaned file (parquet preferred).

    Args:
        input_path: path to CSV file
        output_path: path to save cleaned file; if None, will create sibling file '..._cleaned.parquet'
        save_parquet: whether to save parquet (if False saves csv)

    Returns:
        cleaned DataFrame
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # read CSV (detect encoding issues by using engine)
    try:
        df = pd.read_csv(in_path)
    except Exception:
        df = pd.read_csv(in_path, encoding='utf-8', engine='python')

    review_col, time_col = detect_columns(df)
    if review_col is None:
        raise ValueError('Cannot detect review column in input file')
    if time_col is None:
        # create time_col as NaT if not found
        df['parsed_time'] = pd.NaT
    else:
        df['parsed_time'] = df[time_col].apply(parse_time_column)

    # Clean review text into a new column 'review_content_processed'
    df['review_content_processed'] = df[review_col].apply(clean_text)

    # If time parsed exists, ensure it's datetime and store as 'timestamp'
    if 'parsed_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['parsed_time'], errors='coerce')

    # prepare output path
    if output_path is None:
        out = in_path.parent / (in_path.stem + '_cleaned.parquet')
    else:
        out = Path(output_path)

    if save_parquet:
        # save parquet
        try:
            df.to_parquet(out, index=False)
        except Exception:
            # fallback to csv
            out_csv = out.with_suffix('.csv')
            df.to_csv(out_csv, index=False, encoding='utf-8-sig')
            return df
    else:
        df.to_csv(out, index=False, encoding='utf-8-sig')

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='input CSV file')
    parser.add_argument('--output', '-o', required=False, help='output file (parquet preferred)')
    args = parser.parse_args()
    df_clean = clean_file(args.input, args.output)
    print(df_clean.head().to_string())
