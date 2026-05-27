import os
import re
import sys
import json
import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration & Logging
# -----------------------------
NEWS_DIR = r"C:\Users\12932\Desktop\nus\BAP\data_nolabel\cleaned\news"
COMBINED_REVIEWS_DIR = r"C:\Users\12932\Desktop\nus\BAP\data_nolabel\combined"
OUTPUT_BASE = r"c:\Users\12932\Desktop\nus\BAP\analysis_results\data_summary"

# Only check these specified news columns for content/title quality (no auto-detection)
NEWS_TEXT_COLUMNS: List[str] = [
	'title',
	'contents',
	'title_clean',
	'contents_clean',
]

os.makedirs(OUTPUT_BASE, exist_ok=True)

logging.basicConfig(
	level=logging.INFO,
	format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Helpers: Robust loaders
# -----------------------------

def _parse_datetime_cols(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
	"""Try parse any of candidate columns as datetime in-place; return chosen col or None."""
	for c in candidates:
		if c in df.columns:
			try:
				df[c] = pd.to_datetime(df[c], errors='coerce')
				return c
			except Exception:
				continue
	# fallback: try heuristic by name
	for c in df.columns:
		lc = str(c).lower()
		if 'date' in lc or 'time' in lc:
			try:
				df[c] = pd.to_datetime(df[c], errors='coerce')
				return c
			except Exception:
				continue
	return None


def load_all_news(news_dir: str) -> pd.DataFrame:
	"""
	Load and unify all news/update files into columns ['appid', 'news_date'].
	Supports CSV/XLS/XLSX. Drops rows without valid appid or news_date.
	"""
	rows = []
	for fname in os.listdir(news_dir):
		path = os.path.join(news_dir, fname)
		if not os.path.isfile(path):
			continue
		lower = fname.lower()
		try:
			if lower.endswith('.csv'):
				df = pd.read_csv(path, encoding='utf-8')
			elif lower.endswith(('.xls', '.xlsx')):
				df = pd.read_excel(path, engine='openpyxl')
			else:
				logger.debug(f"Skipping unsupported news file type: {path}")
				continue
		except Exception as e:
			logger.warning(f"Failed to read news file {path}: {e}")
			continue

		# appid
		if 'appid' in df.columns:
			df['appid'] = pd.to_numeric(df['appid'], errors='coerce')
		else:
			m = re.search(r'cleaned[_\-]?(\d+)[_\-]?news', fname, flags=re.IGNORECASE)
			if m:
				df['appid'] = int(m.group(1))
			else:
				df['appid'] = np.nan

		# date
		date_col = _parse_datetime_cols(df, ['post_date', 'date', 'published_at', 'publish_date', 'news_date', 'created_at'])
		if date_col is None:
			df['news_date'] = pd.NaT
		else:
			if date_col != 'news_date':
				df['news_date'] = df[date_col]

		# keep valid
		df = df.loc[df['news_date'].notna()].copy()
		rows.append(df)  # Keep all columns instead of only ['appid', 'news_date']

	if not rows:
		logger.warning(f"No news files loaded from {news_dir}")
		return pd.DataFrame()

	news = pd.concat(rows, ignore_index=True)
	news['appid'] = pd.to_numeric(news['appid'], errors='coerce')
	news = news.dropna(subset=['appid', 'news_date']).copy()
	news['appid'] = news['appid'].astype(int)
	return news


def load_reviews_file(path: str, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
	"""Load a combined reviews file (CSV/XLS/XLSX). Return DataFrame or None on failure."""
	try:
		lower = path.lower()
		if lower.endswith('.csv'):
			return pd.read_csv(path, encoding='utf-8', nrows=max_rows)
		elif lower.endswith(('.xls', '.xlsx')):
			if max_rows is not None:
				return pd.read_excel(path, engine='openpyxl', nrows=max_rows)
			return pd.read_excel(path, engine='openpyxl')
		else:
			logger.debug(f"Skipping unsupported review file type: {path}")
			return None
	except Exception as e:
		logger.warning(f"Failed to read reviews file {path}: {e}")
		return None


def infer_genre_from_filename(fname: str) -> str:
	"""Infer genre label from filenames like combined_fps_reviews.csv; else use stem."""
	m = re.search(r'combined[_\-]?(.+?)[_\-]?reviews', fname, flags=re.IGNORECASE)
	if m:
		return m.group(1).lower()
	return os.path.splitext(fname)[0]


# -----------------------------
# Summaries
# -----------------------------

def summarize_news_per_app(news: pd.DataFrame) -> Dict[int, Dict[str, any]]:
	"""Compute per-app summaries for a unified news DataFrame.

	Returns a mapping: appid -> summary-dict with keys: 'overall', 'monthly',
	'missing_by_col', 'text_content_summary', 'categorical_summary'.
	"""
	out: Dict[int, Dict[str, any]] = {}
	if news.empty:
		return out

	# Ensure news_date is datetime
	news['news_date'] = pd.to_datetime(news['news_date'], errors='coerce')

	# iterate each appid
	for appid, grp in news.groupby('appid'):
		g = grp.copy()
		app_summary: Dict[str, any] = {}
		app_summary['total_rows'] = int(len(g))
		app_summary['appid'] = int(appid)
		app_summary['date_min'] = str(g['news_date'].min()) if g['news_date'].notna().any() else None
		app_summary['date_max'] = str(g['news_date'].max()) if g['news_date'].notna().any() else None

		# monthly counts
		monthly = g.assign(month=g['news_date'].dt.to_period('M')).groupby('month').size().reset_index(name='count')
		monthly['month'] = monthly['month'].astype(str)

		# missingness by column
		missing_by_col = g.isna().mean().sort_values(ascending=False).reset_index()
		missing_by_col.columns = ['column', 'missing_ratio']

		# content/title fields
		reviewed_cols: List[Dict[str, any]] = []
		def _summarize_text_col(df: pd.DataFrame, col: str) -> Dict[str, any]:
			s = df[col]
			missing_ratio = float(s.isna().mean())
			lengths = s.dropna().astype(str).str.len()
			return {
				'column': col,
				'missing_ratio': missing_ratio,
				'len_mean': float(lengths.mean()) if len(lengths) else None,
				'len_median': float(lengths.median()) if len(lengths) else None,
				'len_max': int(lengths.max()) if len(lengths) else None,
			}

		for col in NEWS_TEXT_COLUMNS:
			if col in g.columns:
				reviewed_cols.append(_summarize_text_col(g, col))
		text_content_summary = pd.DataFrame(reviewed_cols) if reviewed_cols else pd.DataFrame(columns=['column','missing_ratio','len_mean','len_median','len_max'])

		# categorical stats (subset)
		categorical_stats = []
		if 'is_update' in g.columns:
			is_update_counts = g['is_update'].value_counts(dropna=False).to_dict()
			categorical_stats.append({
				'column': 'is_update',
				'type': 'boolean',
				'missing_ratio': float(g['is_update'].isna().mean()),
				'value_counts': str(is_update_counts),
			})
		if 'author' in g.columns:
			categorical_stats.append({
				'column': 'author',
				'type': 'categorical',
				'missing_ratio': float(g['author'].isna().mean()),
				'unique_count': int(g['author'].nunique(dropna=True)),
				'top_5_authors': str(g['author'].value_counts().head(5).to_dict()) if g['author'].notna().any() else 'N/A',
			})
		for col in ['url', 'is_external_url', 'feedlabel', 'feedname', 'tags']:
			if col in g.columns:
				val = g[col]
				categorical_stats.append({
					'column': col,
					'type': 'categorical',
					'missing_ratio': float(val.isna().mean()),
					'unique_count': int(val.nunique(dropna=True)) if val.nunique(dropna=True) else 0,
					'sample_values': str(val.dropna().head(3).tolist()) if val.notna().any() else 'N/A',
				})
		categorical_summary = pd.DataFrame(categorical_stats) if categorical_stats else pd.DataFrame(columns=['column', 'type', 'missing_ratio'])

		app_summary.update({
			'monthly': monthly,
			'missing_by_col': missing_by_col,
			'text_content_summary': text_content_summary,
			'categorical_summary': categorical_summary,
		})
		out[int(appid)] = app_summary

	return out


def summarize_reviews_df(df: pd.DataFrame) -> Dict[str, any]:
	"""Compute descriptive stats for a combined reviews DataFrame (per file)."""
	stats: Dict[str, any] = {}
	stats['rows'] = int(len(df))
	stats['cols'] = int(len(df.columns))

	# appid
	if 'appid' in df.columns:
		with np.errstate(all='ignore'):
			appid_num = pd.to_numeric(df['appid'], errors='coerce')
		stats['unique_appids'] = int(appid_num.nunique(dropna=True))
	else:
		stats['unique_appids'] = 0

	# review date range
	review_date_col = None
	for c in ['review_date', 'date', 'created_at', 'timestamp']:
		if c in df.columns:
			try:
				tmp = pd.to_datetime(df[c], errors='coerce')
				review_date_col = c
				stats['review_date_min'] = str(tmp.min()) if tmp.notna().any() else None
				stats['review_date_max'] = str(tmp.max()) if tmp.notna().any() else None
				break
			except Exception:
				continue
	if review_date_col is None:
		stats['review_date_min'] = None
		stats['review_date_max'] = None

	# text length stats
	text_col = None
	for c in ['review_content_clean', 'review_content_processed', 'review_content', 'content', 'text']:
		if c in df.columns:
			text_col = c
			lengths = df[c].astype(str).str.len()
			stats['text_len_mean'] = float(lengths.mean()) if len(lengths) else 0.0
			stats['text_len_median'] = float(lengths.median()) if len(lengths) else 0.0
			stats['text_len_max'] = int(lengths.max()) if len(lengths) else 0
			break

	# sentiment
	if 'vader_compound' in df.columns:
		vc = pd.to_numeric(df['vader_compound'], errors='coerce')
		stats['vader_mean'] = float(vc.mean()) if vc.notna().any() else None
		stats['vader_std'] = float(vc.std()) if vc.notna().any() else None

	# recommendation ratio
	rec_col = 'voted_up' if 'voted_up' in df.columns else ('recommend' if 'recommend' in df.columns else None)
	if rec_col:
		val = df[rec_col]
		if val.dtype == bool:
			pos_ratio = float(val.mean())
		else:
			pos_ratio = float(pd.to_numeric(val, errors='coerce').fillna(0).clip(0, 1).mean())
		stats['recommend_ratio'] = pos_ratio

	# helpfulness / votes
	for c in ['votes_up', 'helpful', 'helpful_yes']:
		if c in df.columns:
			v = pd.to_numeric(df[c], errors='coerce')
			stats[f'{c}_mean'] = float(v.mean()) if v.notna().any() else None
			break

	# overall missingness
	miss_ratio = float(df.isna().mean().mean()) if df.size else 0.0
	stats['missing_overall_ratio'] = miss_ratio

	# top-10 columns by missingness
	miss_by_col = df.isna().mean().sort_values(ascending=False).head(10)
	stats['top_missing_cols'] = miss_by_col.to_dict()

	# top-10 appids by review count (if appid available)
	if 'appid' in df.columns:
		top_apps = pd.to_numeric(df['appid'], errors='coerce').value_counts().head(10)
		stats['top_appids_by_count'] = {int(k): int(v) for k, v in top_apps.items() if not np.isnan(k)}

	return stats


def summarize_reviews_dir(combined_dir: str, output_dir: str) -> pd.DataFrame:
	"""
	Iterate each combined reviews file separately and compute per-file (genre) statistics.
	Save per-genre TXT summaries and return a consolidated DataFrame summary.
	"""
	rows = []
	for fname in os.listdir(combined_dir):
		path = os.path.join(combined_dir, fname)
		if not os.path.isfile(path):
			continue
		df = load_reviews_file(path)
		if df is None or df.empty:
			logger.info(f"Skipping empty/invalid reviews file: {fname}")
			continue

		genre = infer_genre_from_filename(fname)
		stats = summarize_reviews_df(df)
		stats_row = {
			'file': fname,
			'genre': genre,
			**stats,
		}
		rows.append(stats_row)

		# Save per-genre Excel summary
		xlsx_path = os.path.join(output_dir, f"reviews_summary_{genre}_nolabel.xlsx")
		with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
			# summary key-value sheet (exclude nested dicts handled below)
			summary_rows = [
				{"metric": k, "value": v}
				for k, v in stats.items()
				if k not in ("top_missing_cols", "top_appids_by_count")
			]
			pd.DataFrame(summary_rows).to_excel(writer, sheet_name='summary', index=False)

			# top missing columns
			if isinstance(stats.get('top_missing_cols'), dict) and stats['top_missing_cols']:
				df_missing = (
					pd.DataFrame(
						[(k, v) for k, v in stats['top_missing_cols'].items()],
						columns=['column', 'missing_ratio']
					)
					.sort_values('missing_ratio', ascending=False)
				)
				df_missing.to_excel(writer, sheet_name='top_missing_cols', index=False)

			# top appids by count
			if isinstance(stats.get('top_appids_by_count'), dict) and stats['top_appids_by_count']:
				df_topapps = (
					pd.DataFrame(
						[(k, v) for k, v in stats['top_appids_by_count'].items()],
						columns=['appid', 'count']
					)
					.sort_values('count', ascending=False)
				)
				df_topapps.to_excel(writer, sheet_name='top_appids_by_count', index=False)

		logger.info(f"Saved per-genre reviews summary to Excel: {xlsx_path}")

	if not rows:
		return pd.DataFrame()
	return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------

def main():
	logger.info("Starting data summaries...")
	# 1) Per-app news summaries (do not merge news across appids)
	news = load_all_news(NEWS_DIR)
	if news.empty:
		logger.warning(f"No news files loaded from {NEWS_DIR}")
	else:
		per_app_summaries = summarize_news_per_app(news)
		os.makedirs(OUTPUT_BASE, exist_ok=True)
		for appid, summary in per_app_summaries.items():
			xlsx_path = os.path.join(OUTPUT_BASE, f"news_summary_app_{appid}.xlsx")
			with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
				# overall key-values
				records = [{
					'metric': k,
					'value': v
				} for k, v in {
						'total_rows': summary.get('total_rows'),
						'appid': summary.get('appid'),
						'date_min': summary.get('date_min'),
						'date_max': summary.get('date_max')
					}.items()]
				pd.DataFrame(records).to_excel(writer, sheet_name='overall', index=False)
				# other sheets
				summary['monthly'].to_excel(writer, sheet_name='monthly', index=False)
				summary['missing_by_col'].to_excel(writer, sheet_name='missing_by_col', index=False)
				summary['text_content_summary'].to_excel(writer, sheet_name='text_content', index=False)
				summary['categorical_summary'].to_excel(writer, sheet_name='categorical_summary', index=False)
			logger.info(f"Saved news summary for app {appid} to: {xlsx_path}")

	# 2) Per-file (genre) reviews summaries
	# reviews_summary_df = summarize_reviews_dir(COMBINED_REVIEWS_DIR, OUTPUT_BASE)
	# if not reviews_summary_df.empty:
	# 	reviews_summary_path = os.path.join(OUTPUT_BASE, 'reviews_genre_summary.csv')
	# 	reviews_summary_df.to_csv(reviews_summary_path, index=False, encoding='utf-8')
	# 	logger.info(f"Saved consolidated reviews summary: {reviews_summary_path}")
	# else:
	# 	logger.warning("No reviews summaries generated (no valid files or directory empty)")

	logger.info("Data summaries completed.")
	# If a combined news_all file exists, create its summary as well
	try:
		summarize_news_all_if_present()
	except Exception:
		logger.exception("summarize_news_all_if_present() failed")


def _find_news_all(candidates: Optional[List[str]] = None) -> Optional[str]:
	"""Try to locate a news_all.xlsx/csv file in candidate directories.

	Returns the first matching full path or None.
	"""
	if candidates is None:
		candidates = [NEWS_DIR, os.path.dirname(NEWS_DIR), OUTPUT_BASE]

	for base in candidates:
		if not base:
			continue
		# check common names
		for name in ("news_all.xlsx", "news_all.xls", "news_all.csv"):
			p = os.path.join(base, name)
			if os.path.isfile(p):
				return p
	# fallback: search one level deep
	for base in candidates:
		if not os.path.isdir(base):
			continue
		for root, dirs, files in os.walk(base):
			for f in files:
				if f.lower().startswith('news_all') and f.lower().endswith(('.xlsx', '.xls', '.csv')):
					return os.path.join(root, f)
	return None


def summarize_news_all(news_all_path: str, output_dir: str) -> Dict[str, any]:
	"""Load news_all (xlsx/csv) and produce an overall summary report saved to output_dir.

	Returns the stats dict.
	"""
	logger.info(f"Summarizing combined news file: {news_all_path}")
	lower = news_all_path.lower()
	try:
		if lower.endswith('.csv'):
			df = pd.read_csv(news_all_path, encoding='utf-8')
		else:
			df = pd.read_excel(news_all_path, engine='openpyxl')
	except Exception as e:
		logger.exception(f"无法读取 {news_all_path}: {e}")
		return {}

	stats: Dict[str, any] = {}
	stats['rows'] = int(len(df))

	# appid
	if 'appid' in df.columns:
		with np.errstate(all='ignore'):
			appid_num = pd.to_numeric(df['appid'], errors='coerce')
		stats['unique_appids'] = int(appid_num.nunique(dropna=True))
		top_apps = appid_num.value_counts().head(20)
		stats['top_appids_by_count'] = {int(k): int(v) for k, v in top_apps.items() if not np.isnan(k)}
	else:
		stats['unique_appids'] = 0

	# date range detection
	date_col = _parse_datetime_cols(df, ['news_date', 'post_date', 'date', 'published_at', 'publish_date', 'created_at'])
	if date_col:
		try:
			dmin = pd.to_datetime(df[date_col], errors='coerce').min()
			dmax = pd.to_datetime(df[date_col], errors='coerce').max()
			stats['date_min'] = str(dmin) if pd.notna(dmin) else None
			stats['date_max'] = str(dmax) if pd.notna(dmax) else None
		except Exception:
			stats['date_min'] = None
			stats['date_max'] = None
	else:
		stats['date_min'] = None
		stats['date_max'] = None

	# monthly counts using the detected date column (fallback to 'news_date' if present)
	if date_col and date_col in df.columns:
		tmp_dates = pd.to_datetime(df[date_col], errors='coerce')
	elif 'news_date' in df.columns:
		tmp_dates = pd.to_datetime(df['news_date'], errors='coerce')
	else:
		tmp_dates = pd.Series(pd.NaT, index=df.index)

	monthly = (
		pd.DataFrame({'date': tmp_dates})
		.dropna(subset=['date'])
		.assign(month=lambda x: x['date'].dt.to_period('M'))
		.groupby('month')
		.size()
		.reset_index(name='count')
	)
	monthly['month'] = monthly['month'].astype(str)
	# store both records (for JSON-like use) and keep DataFrame for Excel sheet
	stats['monthly_counts'] = monthly.to_dict(orient='records')
	stats['_monthly_df'] = monthly  # temporary carryover for writing to Excel
	
	# is_update distribution
	if 'is_update' in df.columns:
		stats['is_update_counts'] = df['is_update'].value_counts(dropna=False).to_dict()

	# relevance_score stats
	if 'relevance_score' in df.columns:
		rs = pd.to_numeric(df['relevance_score'], errors='coerce')
		stats['relevance_mean'] = float(rs.mean()) if rs.notna().any() else None
		stats['relevance_median'] = float(rs.median()) if rs.notna().any() else None
		stats['relevance_min'] = float(rs.min()) if rs.notna().any() else None
		stats['relevance_max'] = float(rs.max()) if rs.notna().any() else None

	# selected_reason top
	if 'selected_reason' in df.columns:
		stats['top_selected_reasons'] = df['selected_reason'].value_counts().head(20).to_dict()

	# sample headlines and counts
	for col in ('title_clean', 'contents_clean', 'title', 'contents'):
		if col in df.columns:
			stats[f'sample_{col}'] = df[col].dropna().astype(str).head(5).tolist()

	# write a simple Excel report
	os.makedirs(output_dir, exist_ok=True)
	out_xlsx = os.path.join(output_dir, 'news_all_summary.xlsx')
	try:
		with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
			# key-value sheet - exclude large/nested objects and the temporary _monthly_df
			kv = [
				{'metric': k, 'value': v}
				for k, v in stats.items()
				if k not in ('top_appids_by_count', 'top_selected_reasons', '_monthly_df', 'monthly_counts')
			]
			pd.DataFrame(kv).to_excel(writer, sheet_name='summary', index=False)

			# write monthly counts to its own sheet
			if '_monthly_df' in stats and isinstance(stats['_monthly_df'], pd.DataFrame) and not stats['_monthly_df'].empty:
				# drop the temporary key after writing
				stats['_monthly_df'].to_excel(writer, sheet_name='monthly_counts', index=False)
				# keep the JSON-like monthly_counts in stats but remove internal DF
				del stats['_monthly_df']

			if 'top_appids_by_count' in stats and stats['top_appids_by_count']:
				pd.DataFrame([(k, v) for k, v in stats['top_appids_by_count'].items()], columns=['appid', 'count']).to_excel(writer, sheet_name='top_appids', index=False)
			if 'top_selected_reasons' in stats and stats['top_selected_reasons']:
				pd.DataFrame([(k, v) for k, v in stats['top_selected_reasons'].items()], columns=['reason', 'count']).to_excel(writer, sheet_name='top_reasons', index=False)

		logger.info(f"Saved news_all summary to: {out_xlsx}")
	except Exception as e:
		logger.exception(f"Failed to save news_all summary: {e}")

	return stats


# If this module is used as a script for summaries, optionally detect and summarize news_all
def summarize_news_all_if_present():
	path = _find_news_all()
	if path:
		summarize_news_all(path, OUTPUT_BASE)
	else:
		logger.info('No news_all file found to summarize.')


if __name__ == '__main__':
	main()

