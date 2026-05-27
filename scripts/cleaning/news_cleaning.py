import re
import os
import argparse
from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_INPUT_DIR = r"D:\nus\BAP\multi_games_data"
DEFAULT_OUTPUT_DIR = r"D:\nus\BAP\multi_games_data\data\cleaned"

# Keep letters (all unicode letters), digits, common punctuation and whitespace
# Allowed punctuation: . , ? ! ; : ( ) - ' " / % # + = @ &
# We will remove other symbols, control chars, and unusual unicode blocks (like emoji)
ALLOWED_PUNCT = r"\.,\?\!;:\(\)\-\'\"/\%#\+\=\@\&"
# Build a regex that matches any character NOT in letters/digits/whitespace/allowed punctuation
CLEAN_RE = re.compile(rf"[^\w\d\s{ALLOWED_PUNCT}]+", flags=re.UNICODE)

# NOTE: VERSION_PATTERN removed - version-like strings will no longer be used to infer updates

# Keywords that indicate update/patch/news about updates
UPDATE_KEYWORDS = [
    r"update", r"patch", r"hotfix", r"patch notes", r"update notes", r"changelog",
    r"version", r"patches", r"maintenance", r"deploy", r"rollback",
    r"server maintenance", r"update available", r"released", r"released version",
    r"minor update", r"major update", r"hot fix", r'fixes'
]
# Build a safe regex: escape keywords, allow flexible whitespace for multi-word phrases, and require word boundaries
escaped_keywords = [re.escape(kw).replace(r"\ ", r"\\s+") for kw in UPDATE_KEYWORDS]
UPDATE_RE = re.compile(r"\b(?:" + r"|".join(escaped_keywords) + r")\b", flags=re.IGNORECASE)


def _extract_id_from_filename(path: str) -> int | None:
    name = os.path.basename(path)
    m = re.search(r"(\d+)", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def strip_markup(text: str) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    # Remove HTML anchor tags and their inner text entirely (e.g. <a ...>link text</a>)
    s = re.sub(r"<a\b[^>]*>.*?</a>", '', s, flags=re.IGNORECASE | re.DOTALL)
    # Remove other HTML tags but preserve their inner text
    s = re.sub(r"<[^>]+>", '', s)
    # Replace [url=...]text[/url] with text
    s = re.sub(r"\[url=[^\]]+\](.*?)\[/url\]", r"\1", s, flags=re.IGNORECASE | re.DOTALL)
    # Replace [url]link[/url] with link
    s = re.sub(r"\[url\](.*?)\[/url\]", r"\1", s, flags=re.IGNORECASE | re.DOTALL)
    # Remove escaped bracketed sequences like \[ MAPS ] which are often used to show section headers
    s = re.sub(r"\\\s*\[[^\]]+\]", '', s)
    # Remove common block tags like [p], [/p], [list], [/list], [/*], [*]
    s = re.sub(r"\[/?(?:p|list)\]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[\*\]", " ", s)
    s = re.sub(r"\[/\*\]", " ", s)
    # Remove any other [tag]...[/tag] but keep inner text
    s = re.sub(r"\[\w+(?:=[^\]]+)?\](.*?)\[/\w+\]", r"\1", s, flags=re.IGNORECASE | re.DOTALL)
    # Remove any remaining standalone tags like [tag] or [/tag] and any stray brackets
    s = re.sub(r"\[/?[^\]]+\]", "", s)
    s = s.replace('[', ' ').replace(']', ' ')
    # Normalize spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_content_keep_punct(text: str) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    # First strip common BBCode-like markup (e.g., [p], [list], [url=...]text[/url], [/*])
    s = strip_markup(s)
    # Replace undesirable characters with space
    s = CLEAN_RE.sub(' ', s)
    # Normalize spaces
    s = re.sub(r"\s+", ' ', s)
    return s.strip()


def is_update_news(title: str, content: str) -> bool:
    txt = ' '.join(filter(None, [str(title), str(content)])).lower()
    if UPDATE_RE.search(txt):
        return True
    return False


def process_csv(input_path: str, output_path: str = None, min_score: int = 0) -> pd.DataFrame:
    logger.info(f"加载文件: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    logger.info(f"原始行数: {len(df)}")

    # Expect columns 'title' and 'contents' (or 'content') — be flexible
    if 'contents' not in df.columns and 'content' in df.columns:
        df = df.rename(columns={'content': 'contents'})

    if 'title' not in df.columns or 'contents' not in df.columns:
        logger.warning(f"文件 {os.path.basename(input_path)} 缺少 'title' 或 'contents' 列，跳过处理。")
        # return empty DataFrame with same columns for consistency
        return pd.DataFrame(columns=list(df.columns) + ['title_clean', 'contents_clean', 'is_update'])

    # If filename contains appid, filter rows to keep only matching appid
    file_appid = _extract_id_from_filename(input_path)
    if file_appid is not None:
        if 'appid' in df.columns:
            before_appid = len(df)
            df['appid_num'] = pd.to_numeric(df['appid'], errors='coerce')
            df = df[df['appid_num'] == file_appid]
            removed_appid = before_appid - len(df)
            if removed_appid > 0:
                logger.info(f"按文件名ID过滤（appid={file_appid}）去除行数: {removed_appid}")
            df = df.drop(columns=['appid_num'])
        else:
            logger.warning(f"文件 {os.path.basename(input_path)} 不包含 'appid' 列，无法按文件名ID过滤")

    df = df.dropna(subset=['title', 'contents'])
    df = df[df['title'].astype(str).str.strip().str.len() > 0]
    df = df[df['contents'].astype(str).str.strip().str.len() > 0]

    if 'gid' in df.columns:
        before_gid = len(df)
    
        df['gid_str'] = df['gid'].astype(str).str.strip()
        df = df[df['gid_str'].str.match(r'^\d+$')]
        removed_gid = before_gid - len(df)
        if removed_gid > 0:
            logger.info(f"去除 gid 非数字的行: {removed_gid} 条")

        df = df.drop(columns=['gid_str'])
    else:
        logger.warning(f"文件 {os.path.basename(input_path)} 不包含 'gid' 列，无法按 gid 过滤")


    df['title_clean'] = df['title'].apply(lambda t: clean_content_keep_punct(strip_markup(t)))
    df['contents_clean'] = df['contents'].apply(lambda t: clean_content_keep_punct(strip_markup(t)))


    df['is_update'] = df.apply(lambda r: is_update_news(r['title_clean'], r['contents_clean']), axis=1)
    df_filtered = df[df['is_update']].copy()

    logger.info(f"筛选出更新类新闻行数: {len(df_filtered)}")
    # ------------------ 新增: 按日期分组，保留每个日期最相关的一条更新新闻 ------------------
    # 尝试从标题或正文中提取日期，如果没有可用的日期列则标记为 'unknown'
    MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december"
    DATE_PATTERNS = [
        re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),                      # 2025-10-23
        re.compile(r"\b(\d{1,2}[ /.-](?:%s)[ /.-]\d{4})\b" % MONTHS, flags=re.IGNORECASE),  # 23 Oct 2025 or 23 Oct, 2025
        re.compile(r"\b((?:%s)[ /.-]\d{1,2}[, ]+\d{4})\b" % MONTHS, flags=re.IGNORECASE),    # Oct 23, 2025
        re.compile(r"\b(\d{1,2}[ /.-]\d{1,2}[ /.-]\d{2,4})\b")        # 10/23/2025 or 23/10/2025
    ]

    def parse_date_str(s: str):
        s = s.strip()
        for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                continue
        # try to normalize numeric dates like 10/23/25
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            return None

    def extract_date_from_text(text: str):
        if not text:
            return None
        for pat in DATE_PATTERNS:
            m = pat.search(text)
            if m:
                date_str = m.group(1)
                # cleanup commas
                date_str = date_str.replace(',', ' ')
                parsed = parse_date_str(date_str)
                if parsed:
                    return parsed
        return None

    # If there is an explicit date-like column, prefer it
    date_column = None
    for col in df_filtered.columns:
        if col.lower() in ("date", "published", "published_at", "created_at", "pubtime", "time", "timestamp"):
            date_column = col
            break

    # compute a date_key for grouping
    def get_date_key(row):
        # 1) explicit column
        if date_column and pd.notna(row.get(date_column)):
            try:
                d = pd.to_datetime(row.get(date_column), errors='coerce')
                if pd.notna(d):
                    return d.date().isoformat()
            except Exception:
                pass
        # 2) extract from text
        text = ' '.join(filter(None, [str(row.get('title_clean', '')), str(row.get('contents_clean', ''))]))
        extracted = extract_date_from_text(text)
        if extracted:
            return extracted.isoformat()
        # fallback
        return 'unknown'

    def relevance_score(row):
        txt = ' '.join(filter(None, [str(row.get('title_clean', '')), str(row.get('contents_clean', ''))]))
        score = 0
        # title match boosts
        title = str(row.get('title_clean', ''))
        title_matches = len(UPDATE_RE.findall(title))
        if title_matches:
            # stronger boost when keywords appear in title
            score += 20 * title_matches
        # extra boost for exact title phrases
        if re.search(r"\bpatch\s+notes\b|\bupdate\s+notes\b|\bchangelog\b|\breleased\b|\breleased\s+version\b", title, flags=re.IGNORECASE):
            score += 25
        # content matches (count)
        content = str(row.get('contents_clean', ''))
        content_matches = len(UPDATE_RE.findall(content))
        score += content_matches * 3
    # (version-like strings are not used as signals anymore)
        # small boost for presence of words like 'patch notes' exact phrase
        if re.search(r"patch\s+notes|patchnotes|patch notes|update notes|changelog", txt, flags=re.IGNORECASE):
            score += 8
        # penalize obvious non-update promo terms (optional)
        if re.search(r"sale|giveaway|free|discount|preorder|launch event", txt, flags=re.IGNORECASE):
            score -= 5
        # tie-breaker metric: length of content
        score += min(len(content) // 200, 3)
        return score

    # optional: build a human-readable reason for debugging/inspection
    def build_reason(row):
        parts = []
        title = str(row.get('title_clean', ''))
        content = str(row.get('contents_clean', ''))
        tcount = len(UPDATE_RE.findall(title))
        ccount = len(UPDATE_RE.findall(content))
        # vcount removed (version-like strings ignored)
        if tcount:
            parts.append(f"title_matches={tcount}")
        if ccount:
            parts.append(f"content_matches={ccount}")
        if re.search(r"\bpatch\s+notes\b|\bupdate\s+notes\b|\bchangelog\b", title + ' ' + content, flags=re.IGNORECASE):
            parts.append("contains_patch_notes")
        if re.search(r"sale|giveaway|free|discount|preorder|launch event", title + ' ' + content, flags=re.IGNORECASE):
            parts.append("promo_terms")
        return ';'.join(parts) if parts else 'generic'

    if len(df_filtered) == 0:
        logger.info("没有发现更新类新闻，直接返回空结果。")
        return df_filtered

    df_filtered['date_key'] = df_filtered.apply(get_date_key, axis=1)
    df_filtered['relevance_score'] = df_filtered.apply(relevance_score, axis=1)

    # for each date_key keep the highest scoring row; break ties by longer contents_clean
    # Create a temporary numeric column for content length to allow sorting by column name
    df_filtered['contents_len'] = df_filtered['contents_clean'].str.len().fillna(0).astype(int)
    winners = df_filtered.sort_values(['date_key', 'relevance_score', 'contents_len'], ascending=[True, False, False])
    winners = winners.groupby('date_key', as_index=False).first()
    # remove temporary helper column if present
    if 'contents_len' in winners.columns:
        winners = winners.drop(columns=['contents_len'])

    # add a human-readable reason for why the row scored highly
    if not winners.empty:
        winners['selected_reason'] = winners.apply(build_reason, axis=1)

    logger.info(f"按日期保留的记录数: {len(winners)} (原始更新新闻 {len(df_filtered)})")

    # 过滤掉得分低于阈值的记录（如果提供了 min_score）
    if min_score and not winners.empty:
        before_filter = len(winners)
        winners = winners[winners['relevance_score'] >= min_score].copy()
        removed = before_filter - len(winners)
        logger.info(f"剔除低于分数阈值({min_score})的记录: {removed} 条，剩余: {len(winners)} 条")

    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_xlsx = output_path if output_path.lower().endswith('.xlsx') else output_path.replace('.csv', '.xlsx')
        out_csv = output_path if output_path.lower().endswith('.csv') else output_path.replace('.xlsx', '.csv')
        try:
            winners.to_excel(out_xlsx, index=False, engine='openpyxl')
            logger.info(f"已保存按日期精选的更新新闻到: {out_xlsx}")
        except Exception as e:
            winners.to_csv(out_csv, index=False, encoding='utf-8', quoting=1)
            logger.warning(f"无法保存为 Excel，已回退保存为 CSV: {out_csv}，错误: {e}")

    # 返回按日期精选后的 DataFrame
    return winners


def aggregate_all(cleaned_dir: str, save_path: str | None = None) -> pd.DataFrame:
    """
    汇总一个目录下所有已清洗的单个游戏文件（cleaned_*.xlsx 或 cleaned_*.csv）

    参数:
    - cleaned_dir: 包含 cleaned_*.xlsx/csv 文件的目录
    - save_path: 如果提供，则将合并结果保存为该路径（xlsx 或 csv，根据后缀决定）

    返回合并后的 DataFrame（news_all）
    """
    if not os.path.isdir(cleaned_dir):
        logger.error(f"提供的路径不是目录: {cleaned_dir}")
        return pd.DataFrame()

    files = [f for f in os.listdir(cleaned_dir) if f.lower().startswith('cleaned_') and (f.lower().endswith('.xlsx') or f.lower().endswith('.csv'))]
    if not files:
        logger.info(f"在目录中未找到任何 cleaned_*.xlsx 或 cleaned_*.csv 文件: {cleaned_dir}")
        return pd.DataFrame()

    parts = []
    for f in files:
        path = os.path.join(cleaned_dir, f)
        try:
            if f.lower().endswith('.xlsx'):
                df = pd.read_excel(path, engine='openpyxl')
            else:
                df = pd.read_csv(path, encoding='utf-8')
            parts.append(df)
            logger.info(f"已读取: {f} (rows={len(df)})")
        except Exception as e:
            logger.exception(f"读取已清洗文件 {f} 时出错，已跳过: {e}")

    if not parts:
        logger.info("没有有效的已清洗文件可用于合并。")
        return pd.DataFrame()

    news_all = pd.concat(parts, ignore_index=True)
    # 尝试去重（基于 title_clean + contents_clean + date_key if 存在）
    dedupe_cols = [c for c in ('title_clean', 'contents_clean', 'date_key') if c in news_all.columns]
    if dedupe_cols:
        before = len(news_all)
        news_all = news_all.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
        logger.info(f"合并并去重：原始行 {before} -> 去重后 {len(news_all)}")
    else:
        logger.info(f"合并完成，行数: {len(news_all)}")

    if save_path:
        try:
            if save_path.lower().endswith('.xlsx'):
                news_all.to_excel(save_path, index=False, engine='openpyxl')
            else:
                news_all.to_csv(save_path, index=False, encoding='utf-8')
            logger.info(f"已保存合并结果到: {save_path}")
        except Exception as e:
            logger.exception(f"保存合并结果失败: {e}")

    return news_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='清洗新闻 CSV，只保留游戏更新/补丁相关的新闻')
    parser.add_argument('input', nargs='?', help='输入 CSV 文件或目录 (可选，默认使用配置中的 multi_games_data)')
    parser.add_argument('--out', '-o', help='输出文件或输出目录（目录时会为每个输入文件生成 cleaned_<filename>）', default=None)
    parser.add_argument('--min-score', type=int, default=7, help='剔除低于此相关性分数的记录（默认0，不剔除）')
    args = parser.parse_args()

    inp = args.input or DEFAULT_INPUT_DIR
    out = args.out or DEFAULT_OUTPUT_DIR

    if os.path.isdir(inp):
        files = [f for f in os.listdir(inp) if f.lower().endswith('.csv') and 'news' in f.lower()]
        # Determine output directory for aggregate: prefer provided out if it's a dir, otherwise use inp
        aggregate_dir = out if out and os.path.isdir(out) else inp
        for f in files:
            inpath = os.path.join(inp, f)
            if out and os.path.isdir(out):
                outpath = os.path.join(out, f'cleaned_{f.replace(".csv", ".xlsx")}')
            elif out:
                outpath = os.path.join(os.path.dirname(inpath), f'cleaned_{f.replace(".csv", ".xlsx")}')
            else:
                outpath = os.path.join(os.path.dirname(inpath), f'cleaned_{f.replace(".csv", ".xlsx")}')
            try:
                process_csv(inpath, outpath, min_score=args.min_score)
            except Exception as e:
                logger.exception(f"处理文件 {f} 时发生错误，已跳过: {e}")

        # after processing all files, aggregate cleaned files in aggregate_dir
        try:
            aggregate_all(aggregate_dir, save_path=os.path.join(aggregate_dir, 'news_all.xlsx'))
        except Exception as e:
            logger.exception(f"合并已清洗文件时出错: {e}")
    else:
        if out and os.path.isdir(out):
            outpath = os.path.join(out, f'cleaned_{os.path.basename(inp).replace(".csv", ".xlsx")}')
        elif out:
            outpath = out
        else:
            outpath = os.path.join(os.path.dirname(inp), f'cleaned_{os.path.basename(inp).replace(".csv", ".xlsx")}')
        # process single input file
        process_csv(inp, outpath, min_score=args.min_score)
        # aggregate the single output directory
        try:
            aggregate_all(os.path.dirname(outpath), save_path=os.path.join(os.path.dirname(outpath), 'news_all.xlsx'))
        except Exception as e:
            logger.exception(f"合并已清洗文件时出错: {e}")