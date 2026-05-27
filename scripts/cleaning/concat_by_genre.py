import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GENRE_MAP = {
    'FPS': ['730', '1938090'],
    'leisure': ['413150', '945360'],
    'strategy': ['281990', '289070'],
}


def find_cleaned_file(cleaned_dir: str, steam_id: str):
    candidates = [
        f"cleaned_{steam_id}_reviews.xlsx",
        f"cleaned_{steam_id}_reviews.xls",
        f"cleaned_{steam_id}_reviews.xlsm",
    ]
    for fn in candidates:
        path = os.path.join(cleaned_dir, fn)
        if os.path.exists(path):
            return path
    return None


def concat_by_genre(cleaned_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    for genre, steam_ids in GENRE_MAP.items():
        parts = []
        files_read = []
        for sid in steam_ids:
            path = find_cleaned_file(cleaned_dir, sid)
            if not path:
                logger.warning(f"未找到已清洗文件: cleaned_{sid}_reviews.xlsx 在 {cleaned_dir}")
                continue

            try:
                if path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                    df = pd.read_excel(path, sheet_name=0)
                else:
                    logger.warning(f"跳过非 Excel 文件 {path}")
                    continue
                if 'appid' not in df.columns:
                    df['appid'] = sid
                else:
                    df['appid'] = df['appid'].fillna(sid)
                cols = list(df.columns)
                if cols and cols[0] != 'appid':
                    if 'appid' in cols:
                        cols.remove('appid')
                        cols = ['appid'] + cols
                        df = df[cols]

            except Exception as e:
                logger.warning(f"读取文件失败 {path}: {e}")
                continue

            parts.append(df)
            files_read.append(path)
            logger.info(f"读取 {path}，行数: {len(df)}")

        if not parts:
            logger.info(f"{genre} 没有可合并的文件，跳过。")
            summary[genre] = {'files': 0, 'rows': 0}
            continue

        combined = pd.concat(parts, ignore_index=True, sort=False)
        out_path = os.path.join(out_dir, f"combined_{genre.lower()}_reviews.xlsx")
        try:
            combined.to_excel(out_path, index=False)
            logger.info(f"已保存合并文件 {out_path}，包含 {len(combined)} 行，来源文件: {len(files_read)}")
        except Exception as e:
            logger.error(f"保存合并 Excel 文件失败 {out_path}: {e}")
        
        summary[genre] = {'files': len(files_read), 'rows': len(combined), 'paths': files_read}

    return summary


def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    cleaned_dir = os.path.join(script_dir, 'data_label', 'cleaned', 'reviews')
    out_dir = os.path.join(script_dir, 'data_label', 'combined')

    logger.info(f"从 {cleaned_dir} 读取已清洗文件并按类型合并（输出到 {out_dir}）")
    summary = concat_by_genre(cleaned_dir, out_dir)
    logger.info(f"合并完成。汇总: {summary}")


if __name__ == '__main__':
    main()
