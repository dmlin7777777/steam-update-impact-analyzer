import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timezone
from razdel import tokenize as razdel_tokenize
from konlpy.tag import Okt
try:
    from pythainlp import word_tokenize as thai_tokenize
except Exception:
    thai_tokenize = None
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple, Set
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    
    def __init__(self):
        self.setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        
    def setup_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        self.english_stopwords = set(stopwords.words('english'))
        
    def normalize_language(self, lang: str) -> str:
        if not lang or pd.isna(lang):
            return 'english'
        l = str(lang).strip().lower()
        mapping = {
            'english': 'english', 'ukenglish': 'english', 'usenglish': 'english', 'australian english': 'english',
        }
        return mapping.get(l, l)
    
    def is_spam_content(self, content: str) -> bool:
        if not content or len(content.strip()) < 10:
            return True

        if self._looks_like_checkbox_template(content):
            return True
        
        if re.search(r'(.)\1{10,}', content):
            return True
        
        content_no_checks = re.sub(r'[☐☑✓✔✗✘]', '', content)
        special_char_ratio = len(re.findall(r'[^\w\s\u4e00-\u9fff]', content_no_checks)) / max(len(content_no_checks), 1)
        if special_char_ratio > 0.4:
            return True
        
        spam_keywords = [
            'spam', 'fake', 'bot', 'advertisement', 'buy now', 'click here', 
            'free money', 'earn money', 'make money', 'get rich', 'visit my', 
            'check out', 'follow me', 'subscribe', 'like and subscribe',
            'promotional', 'discount code', 'coupon', 'limited time',
            'act now', 'hurry up', 'dont miss', 'exclusive offer'
        ]
        
        content_lower = content.lower()
        spam_count = sum(1 for keyword in spam_keywords if keyword in content_lower)
        if spam_count >= 2:
            return True
        
        return False

    def _looks_like_checkbox_template(self, text: str) -> bool:
        if not text:
            return False
        has_section = re.search(r"^-+\{\s*[^}]+\s*\}-+\s*$", text, flags=re.MULTILINE) is not None
        has_checkbox = re.search(r"^[\s]*[☐☑]", text, flags=re.MULTILINE) is not None
        return has_section and has_checkbox
    
    def is_bot_review(self, row: pd.Series) -> bool:
        
        played_hours = row.get('played_hours', 0)
        num_reviews = row.get('num_reviews', 0)
        
        if played_hours < 1 and num_reviews > 10:
            return True
        
        content = row.get('review_content', '')
        if len(content.strip()) < 20:
            return True
        
        template_patterns = [
            r'^[\+\-\☐\☑\✓\×]{5,}',  
            r'Graphics.*Audio.*Gameplay', 
            r'^\d+/10', 
        ]
        
        for pattern in template_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True
        
        return False
    
    def clean_text(self, text: str, language: str = 'english') -> str:

        if not text or pd.isna(text):
            return ""
        text = str(text)
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', '', text)
        text = re.sub(r'[，。；：“”‘’、·]', ' ', text)
        text = re.sub(r'[０-９]', lambda m: str(ord(m.group(0)) - ord('０')), text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\uFFFD\u200B\u202A-\u202E]', '', text)
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'(.)\1{6,}', r'\1', text)
        text = re.sub(r'[^\w\s.,!?;:()"\'-]', '', text)
        text = text.strip()
        return text
    
    def remove_stopwords(self, text: str, language: str = 'english') -> str:
        if not text:
            return ""
        try:
            words = word_tokenize(text.lower())
            words = [self.lemmatizer.lemmatize(word) for word in words 
                    if word not in self.english_stopwords and word.isalpha() and len(word) > 2]
            return ' '.join(words)
        except Exception:
            words = text.lower().split()
            words = [word for word in words if word not in self.english_stopwords and len(word) > 2]
            return ' '.join(words)
    
    def clean_reviews_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"开始清洗评论数据，原始数据量: {len(df)}")
      
        cleaned_df = df.copy()

        cleaned_df = cleaned_df.drop_duplicates()
        cleaned_df = cleaned_df.dropna(how='all')
        cleaned_df.reset_index(drop=True, inplace=True)

        cleaned_df = cleaned_df.dropna(subset=['review_content'])
        cleaned_df = cleaned_df[cleaned_df['review_content'].astype(str).str.strip().str.len() > 0]
        logger.info(f"去除缺失或空评论内容后: {len(cleaned_df)}")

        if 'language' in cleaned_df.columns:
            cleaned_df['language_norm'] = cleaned_df['language'].apply(self.normalize_language)
            cleaned_df = cleaned_df[cleaned_df['language_norm'] == 'english']
        else:
            cleaned_df['language_norm'] = 'english'
        logger.info(f"只保留英语评论后: {len(cleaned_df)}")

        cleaned_df['review_content_clean'] = cleaned_df.apply(
            lambda row: self.clean_text(row['review_content'], 'english'), axis=1
        )
        cleaned_df = cleaned_df[cleaned_df['review_content_clean'].str.strip().str.len() > 0]
        logger.info(f"文本清洗后剩余: {len(cleaned_df)}")

        # 初始化标记列：0 表示保留，1 表示被清洗规则（垃圾/机器人）标记
        cleaned_df['is_removed_cleaning'] = 0

        # 标记垃圾评论（不直接删除）
        spam_mask = cleaned_df['review_content'].apply(self.is_spam_content)
        cleaned_df.loc[spam_mask, 'is_removed_cleaning'] = 1
        logger.info(f"标记垃圾评论: {spam_mask.sum()} 条（已保留、标记为 is_removed_cleaning=1）")

        # 机器人检测：使用原来的规则，但不删除行，只标记
        mask_bot = (
            (cleaned_df['played_hours'].fillna(0) < 1) & (cleaned_df['num_reviews'].fillna(0) > 10)
        )
        mask_short = cleaned_df['review_content'].str.strip().str.len() < 10
        mask_template_symbol = cleaned_df['review_content'].str.contains(r'^[\+\-\☐\☑\✓\×]{5,}$', regex=True)
        mask_template_score = cleaned_df['review_content'].str.contains(r'^\d+/10$', regex=True)
        mask_template_remove = (mask_template_symbol | mask_template_score) & mask_short
        mask_all = mask_bot | mask_template_remove
        cleaned_df.loc[mask_all, 'is_removed_cleaning'] = 1
        logger.info(f"标记机器人评论: {mask_all.sum()} 条（已保留、标记为 is_removed_cleaning=1）")

        # 去除过短评论：不再保留此前被标记为 is_removed_cleaning 的行
        # 仅保留长度 >= 20 的评论
        length_mask = cleaned_df['review_content'].str.strip().str.len() >= 20
        cleaned_df = cleaned_df[length_mask]
        logger.info(f"去除过短评论后: {len(cleaned_df)}")

        cleaned_df['review_content_processed'] = cleaned_df.apply(
            lambda row: self.remove_stopwords(row['review_content_clean'], 'english'), axis=1
        )
        cleaned_df = cleaned_df[cleaned_df['review_content_processed'].str.strip().str.len() > 0]
        logger.info(f"去除停用词后剩余: {len(cleaned_df)}")

        essential_cols = ['review_content', 'review_content_clean', 'review_content_processed']
        for col in essential_cols:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.dropna(subset=[col])
                cleaned_df = cleaned_df[cleaned_df[col].astype(str).str.strip().str.len() > 0]

        logger.info(f"最终清洗后数据量: {len(cleaned_df)}")

        return cleaned_df
        
def main():
    logger.info("开始数据清洗和预处理")
    
    cleaner = DataCleaner()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = r'D:\nus\BAP\multi_games_data'
    base_output_dir = r'D:\nus\BAP\multi_games_data'
    # ensure base output dir exists; per-file-type subdirs will be created as needed
    os.makedirs(base_output_dir, exist_ok=True)
    
    for filename in os.listdir(data_dir):
        lower_name = filename.lower()
        if lower_name.endswith(('.xlsx', '.csv')):
            file_path = os.path.join(data_dir, filename)
            logger.info(f"处理文件: {filename}")

            if 'review' in lower_name:
                output_dir = os.path.join(base_output_dir, 'reviews')
            elif 'news' in lower_name:
                output_dir = os.path.join(base_output_dir, 'news')
            else:
                output_dir = os.path.join(base_output_dir, 'other')
            os.makedirs(output_dir, exist_ok=True)
            
            if lower_name.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"文件 {filename} 为空或无法读取")
                continue
            
            cleaner._current_filename = os.path.join(output_dir, filename)
            
            # Note: SteamID numeric filtering will be applied after cleaning to avoid
            # accidentally removing rows required for language detection or other steps.
            # Detect steam id column name (case-insensitive) so we can apply it later.
            steam_cols = [c for c in df.columns if c.lower() in (
                'steamid', 'steam_id', 'steam id', 'steamid64', 'authorid', 'author id', 'author_id')]
            steam_col = steam_cols[0] if steam_cols else None

    
            if 'review' in lower_name:
    
                removed_rows = pd.DataFrame()
                cleaned_df = cleaner.clean_reviews_data(df)
                # After cleaning, drop rows where SteamID column is not strictly numeric
                if steam_col and steam_col in cleaned_df.columns:
                    before_rows = len(cleaned_df)
                    cleaned_df[steam_col] = cleaned_df[steam_col].astype(str).str.strip()
                    mask_numeric = cleaned_df[steam_col].str.match(r'^\d+$')
                    removed_mask = ~mask_numeric
                    removed = removed_mask.sum()
                    if removed > 0:
                        removed_rows = cleaned_df[removed_mask].copy()
                        logger.info(f"在文件 {filename} 中移除 {removed} 行：{steam_col} 不是纯数字（在清洗后）")
                    cleaned_df = cleaned_df[mask_numeric].copy()
                elif steam_col:
                    logger.debug(f"在清洗后未找到列 {steam_col}，跳过 SteamID 过滤")
            else:
                logger.info(f"跳过非评论文件: {filename}")
                continue
            
            if 'review_content' in cleaned_df.columns:
                before_final = len(cleaned_df)
                cleaned_df = cleaned_df[cleaned_df['review_content'].astype(str).str.strip().str.len() > 0]
                if before_final != len(cleaned_df):
                    logger.info(f"保存前最终去除空评论: {before_final - len(cleaned_df)} 条")
            
            name_no_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"cleaned_{name_no_ext}.xlsx")

            def _test_write(df_part: pd.DataFrame) -> bool:
                buf = BytesIO()
                try:
                    df_part.to_excel(buf, index=False, engine='openpyxl')
                    return True
                except Exception:
                    return False

            def _find_bad_indices(df_part: pd.DataFrame) -> List:
                bad = []
                idx_list = list(df_part.index)

                def _recurse(idxs: List):
                    if not idxs:
                        return
                    part = df_part.loc[idxs]
                    if _test_write(part):
                        return
                    if len(idxs) == 1:
                        bad.append(idxs[0])
                        return
                    mid = len(idxs) // 2
                    _recurse(idxs[:mid])
                    _recurse(idxs[mid:])

                _recurse(idx_list)
                return bad

            removed_by_excel = pd.DataFrame()
            attempt_df = cleaned_df
            while True:
                try:
                    attempt_df.to_excel(output_path, index=False, engine='openpyxl')
                    logger.info(f"已保存清洗后的数据到: {output_path}")
                    break
                except Exception as e:
                    logger.warning(f"写入 Excel 失败，尝试定位并移除出错行: {e}")
                    bad_idxs = _find_bad_indices(attempt_df)
                    if not bad_idxs:
                        logger.error("无法定位出错行，写入 Excel 失败且没有找到具体行，停止保存。")
                        raise
                    # collect removed rows
                    bad_rows = attempt_df.loc[bad_idxs].copy()
                    removed_by_excel = pd.concat([removed_by_excel, bad_rows], axis=0) if not removed_by_excel.empty else bad_rows
                    # drop them and retry
                    attempt_df = attempt_df.drop(index=bad_idxs)
                    logger.info(f"已移除 {len(bad_idxs)} 行后重试写入 Excel")

            if 'removed_rows' in locals() and isinstance(removed_rows, pd.DataFrame) and not removed_by_excel.empty:
                removed_rows = pd.concat([removed_rows, removed_by_excel], axis=0)
            elif not removed_by_excel.empty:
                removed_rows = removed_by_excel
            
            report_path = os.path.join(output_dir, f"cleaning_report_{name_no_ext}.txt")
            flagged_total = int(cleaned_df['is_removed_cleaning'].fillna(0).astype(int).sum()) if 'is_removed_cleaning' in cleaned_df.columns else 0
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"数据清洗报告 - {filename}\n")
                f.write("=" * 50 + "\n")
                f.write(f"原始数据量: {len(df)}\n")
                f.write(f"清洗后数据量: {len(cleaned_df)}\n")
                f.write(f"被标记为清洗的行数 (is_removed_cleaning==1)：{flagged_total}\n")
                f.write(f"清洗比例: {(len(df) - len(cleaned_df)) / len(df) * 100:.2f}%\n")
                f.write(f"清洗时间: {datetime.now()}\n")

            removed_rows_path_xlsx = os.path.join(output_dir, f"removed_rows_{name_no_ext}.xlsx")
            removed_rows_path_csv = os.path.join(output_dir, f"removed_rows_{name_no_ext}.csv")
            try:
                flagged_rows = pd.DataFrame()
                if 'is_removed_cleaning' in cleaned_df.columns:
                    flagged_rows = cleaned_df[cleaned_df['is_removed_cleaning'].fillna(0).astype(int) == 1].copy()

                combined_removed = pd.DataFrame()
                if 'removed_rows' in locals() and isinstance(removed_rows, pd.DataFrame) and not removed_rows.empty:
                    combined_removed = removed_rows.copy()

                if not flagged_rows.empty:
                    combined_removed = pd.concat([combined_removed, flagged_rows], axis=0) if not combined_removed.empty else flagged_rows

                if not combined_removed.empty:
                    try:
                        combined_removed.to_excel(removed_rows_path_xlsx, index=False, engine='openpyxl')
                        logger.info(f"已保存被标记/被移除的行到: {removed_rows_path_xlsx}")
                    except Exception as e:
                        try:
                            combined_removed.to_csv(removed_rows_path_csv, index=False, encoding='utf-8', quoting=1)
                            logger.warning(f"无法保存为 Excel，已回退保存被移除/标记行为 CSV: {removed_rows_path_csv}，错误: {e}")
                        except Exception:
                            logger.exception("保存被移除/标记行时发生错误")
                else:
                    logger.debug("此次清洗未生成单独的 removed_rows/flagged_rows DataFrame 或其为空，跳过保存被移除行的文件")
            except Exception as e:
                try:
                    if 'removed_rows' in locals() and isinstance(removed_rows, pd.DataFrame) and not removed_rows.empty:
                        removed_rows.to_csv(removed_rows_path_csv, index=False, encoding='utf-8', quoting=1)
                        logger.warning(f"无法保存为 Excel，已回退保存被移除行为 CSV: {removed_rows_path_csv}，错误: {e}")
                except Exception:
                    logger.exception("保存被移除行时发生错误")
    
    logger.info("数据清洗和预处理完成!")

if __name__ == "__main__":
    main()