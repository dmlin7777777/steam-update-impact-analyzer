import requests
import pandas as pd
import time
import os
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:
    try:
        from urllib3.util import Retry
    except ImportError:
        from requests.adapters import DEFAULT_RETRIES as Retry

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SteamScraper:
    def __init__(self, delay: float = 0.5, max_workers: int = 3):
        """
        初始化Steam爬虫
        
        Args:
            delay: 请求间隔时间（秒），建议1-2秒以遵守robots协议
            max_workers: 最大并发线程数，建议不超过3
        """
        self.delay = delay
        self.max_workers = max_workers
        self.session = self._create_session()
        self.headers = {
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Connection': 'keep-alive'
        }
        
        # 检查robots.txt
        self._check_robots_txt()
    
    def _create_session(self) -> requests.Session:
        """创建带有重试策略的session"""
        session = requests.Session()
        
        # 设置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _check_robots_txt(self):
        """检查并遵守robots.txt协议"""
        try:
            # 检查Steam Store robots.txt
            store_rp = urllib.robotparser.RobotFileParser()
            store_rp.set_url("https://store.steampowered.com/robots.txt")
            store_rp.read()
            
            # 检查Steam Community robots.txt
            community_rp = urllib.robotparser.RobotFileParser()
            community_rp.set_url("https://steamcommunity.com/robots.txt")
            community_rp.read()
            
            # 检查API访问是否被允许
            api_url = "https://store.steampowered.com/appreviews/"
            news_api_url = "https://api.steampowered.com/ISteamNews/"
            
            if not store_rp.can_fetch(self.headers['User-Agent'], api_url):
                logger.warning("API访问可能受robots.txt限制，请谨慎使用")
            
            logger.info("Robots.txt检查完成")
            
        except Exception as e:
            logger.warning(f"无法检查robots.txt: {e}")
    
    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """发起请求并处理错误"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                # 遵守速率限制
                time.sleep(self.delay)
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"请求最终失败: {url}")
                    return None
    
    def get_reviews_batch(self, game_id: str, cursor: str = '*', batch_size: int = 100) -> tuple[List[Dict], str]:
        """批量获取评论数据"""
        url = f'https://store.steampowered.com/appreviews/{game_id}'
        params = {
            'json': '1',
            'filter': 'recent',
            'language': 'english',
            'cursor': cursor,
            'purchase_type': 'all',
            'num_per_page': min(batch_size, 100)  # Steam API限制最多100条
        }
        
        data = self._make_request(url, params)
        if not data or not data.get('reviews'):
            return [], cursor
        
        reviews = []
        for review in data['reviews']:
            review_entry = {
                'SteamID': review['author']['steamid'],
                'played_hours': review['author']['playtime_forever'],
                'num_games_owned': review['author'].get('num_games_owned', 'Private Account') or 'Private Account',
                'num_reviews': review['author']['num_reviews'],
                'playtime_last_two_weeks': review['author']['playtime_last_two_weeks'],
                'playtime_at_review': review['author']['playtime_at_review'] if review['author'].get('playtime_at_review') is not None else 'Unknown',
                'last_played': time.strftime("%Y-%m-%d", time.gmtime(review['author']['last_played'])) if review['author'].get('last_played') else '',
                'language': review['language'],
                'review_content': review['review'],
                'voted_up': 1 if review['voted_up'] else 0,
                'review_date': time.strftime("%Y-%m-%d", time.gmtime(review.get('timestamp_created', 0))),
                'last_updated': time.strftime("%Y-%m-%d", time.gmtime(review.get('timestamp_updated', 0))),
                'votes_up': review['votes_up'],
                'votes_funny': review['votes_funny'],
                'weighted_vote_score': review['weighted_vote_score'],
                'comment_count': review['comment_count'],
                'steam_purchase': 1 if review['steam_purchase'] else 0,
                'received_for_free': 1 if review['received_for_free'] else 0,
                'written_during_early_access': 1 if review['written_during_early_access'] else 0,
                'primarily_steam_deck': 1 if review['primarily_steam_deck'] else 0
            }
            reviews.append(review_entry)
        
        return reviews, data.get('cursor', cursor)
    
    def scrape_reviews(self, game_id: str, max_reviews: int = 60000, output_file: str = None, save_progress: bool = False) -> pd.DataFrame:
        """
        爬取游戏评论
        
        Args:
            game_id: 游戏ID
            max_reviews: 最大评论数量限制，None表示爬取所有评论
            output_file: 输出文件路径
            save_progress: 是否定期保存进度（每1000条保存一次），默认关闭
        """
        logger.info(f"开始爬取游戏 {game_id} 的评论数据 {'（所有评论）' if max_reviews is None else f'（最多{max_reviews}条）'}")
        
        all_reviews = []
        cursor = '*'
        page = 1
        last_save_count = 0
        consecutive_empty_pages = 0
        max_empty_pages = 5  # 允许最多5页连续空页面
        
        while True:
            logger.info(f"正在获取第 {page} 页数据...")
            
            reviews, new_cursor = self.get_reviews_batch(game_id, cursor)
            
            # 检查是否有新数据
            if not reviews:
                consecutive_empty_pages += 1
                logger.warning(f"第 {page} 页无数据，连续空页面: {consecutive_empty_pages}")
                if consecutive_empty_pages >= max_empty_pages:
                    logger.info(f"连续 {max_empty_pages} 页无数据，停止爬取")
                    break
            else:
                consecutive_empty_pages = 0
                all_reviews.extend(reviews)
                
            # 检查游标是否变化
            if cursor == new_cursor:
                logger.info("游标未变化，已获取所有数据")
                break
                
            cursor = new_cursor
            page += 1
            
            logger.info(f"已获取 {len(reviews)} 条评论，总计 {len(all_reviews)} 条")
            
            
            # 检查是否达到最大数量限制
            if max_reviews and len(all_reviews) >= max_reviews:
                all_reviews = all_reviews[:max_reviews]
                logger.info(f"达到最大限制 {max_reviews} 条评论")
                break
        
        logger.info(f"评论爬取完成！共获取 {len(all_reviews)} 条评论")
        df = pd.DataFrame(all_reviews)
        
        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"最终数据已保存到 {output_file}")
            
            # 删除临时进度文件
            if save_progress:
                import glob
                progress_files = glob.glob(output_file.replace('.csv', '_progress_*.csv'))
                for pf in progress_files:
                    try:
                        os.remove(pf)
                    except:
                        pass
        
        return df
    
    def scrape_news(self, game_id: str, output_file: str = None, max_news: int = None) -> pd.DataFrame:
        """
        爬取游戏新闻
        
        Args:
            game_id: 游戏ID
            output_file: 输出文件路径
            max_news: 最大新闻数量限制，None表示爬取所有新闻
        """
        logger.info(f"开始爬取游戏 {game_id} 的新闻数据 {'（所有新闻）' if max_news is None else f'（最多{max_news}条）'}")
        
        # 首先获取新闻总数
        url = f'https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/'
        params = {'appid': game_id, 'count': 1}
        
        data = self._make_request(url, params)
        if not data or 'appnews' not in data:
            logger.error("无法获取新闻数据")
            return pd.DataFrame()
        
        total_count = data['appnews']['count']
        logger.info(f"发现 {total_count} 条新闻")
        
        # 确定实际要获取的新闻数量
        actual_count = total_count
        if max_news is not None and max_news < total_count:
            actual_count = max_news
            logger.info(f"限制获取数量为 {actual_count} 条")
        
        # Steam API可能有单次请求数量限制，分批获取
        max_per_request = 1000  # Steam API建议的单次最大请求数
        all_news_data = []
        
        for offset in range(0, actual_count, max_per_request):
            batch_count = min(max_per_request, actual_count - offset)
            
            logger.info(f"正在获取第 {offset + 1} 到第 {offset + batch_count} 条新闻...")
            
            # 获取这一批新闻
            params = {
                'appid': game_id, 
                'count': batch_count,
                'maxlength': 0  # 获取完整内容
            }
            
            data = self._make_request(url, params)
            
            if not data or 'appnews' not in data:
                logger.warning(f"获取第 {offset + 1} 批新闻失败")
                continue
            
            batch_news = data['appnews']['newsitems']
            
            for news in batch_news:
                try:
                    news_entry = {
                        'gid': news.get('gid', ''),
                        'title': news.get('title', ''),
                        'url': news.get('url', ''),
                        'is_external_url': news.get('is_external_url', False),
                        'author': news.get('author', ''),
                        'contents': news.get('contents', ''),
                        'feedlabel': news.get('feedlabel', ''),
                        'date': time.strftime("%Y-%m-%d", time.gmtime(news.get('date', 0))) if news.get('date') else '',
                        'feedname': news.get('feedname', ''),
                        'appid': news.get('appid', game_id),
                        'tags': ','.join(news.get('tags', [])) if news.get('tags') else ''
                    }
                    all_news_data.append(news_entry)
                except Exception as e:
                    logger.warning(f"解析新闻条目时出错: {e}")
                    continue
            
            logger.info(f"已获取 {len(batch_news)} 条新闻，总计 {len(all_news_data)} 条")
            
            # 如果这批数据不足，说明已经获取完所有数据
            if len(batch_news) < batch_count:
                logger.info("已获取所有可用新闻数据")
                break
        
        logger.info(f"新闻爬取完成！共获取 {len(all_news_data)} 条新闻")
        df = pd.DataFrame(all_news_data)
        
        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"新闻数据已保存到 {output_file}")
        
        return df
    
    def scrape_multiple_games(self, game_ids: List[str], base_output_dir: str = None) -> Dict[str, Dict]:
        """并发爬取多个游戏的数据"""
        logger.info(f"开始并发爬取 {len(game_ids)} 个游戏的数据")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交评论爬取任务
            review_futures = {
                executor.submit(
                    self.scrape_reviews, 
                    game_id, 
                    output_file=f"{base_output_dir}/{game_id}_reviews.csv" if base_output_dir else None
                ): game_id for game_id in game_ids
            }
            
            # 提交新闻爬取任务
            news_futures = {
                executor.submit(
                    self.scrape_news, 
                    game_id, 
                    output_file=f"{base_output_dir}/{game_id}_news.csv" if base_output_dir else None
                ): game_id for game_id in game_ids
            }
            
            # 处理评论结果
            for future in as_completed(review_futures):
                game_id = review_futures[future]
                try:
                    df = future.result()
                    if game_id not in results:
                        results[game_id] = {}
                    results[game_id]['reviews'] = df
                    logger.info(f"游戏 {game_id} 评论爬取完成")
                except Exception as e:
                    logger.error(f"游戏 {game_id} 评论爬取失败: {e}")
            
            # 处理新闻结果
            for future in as_completed(news_futures):
                game_id = news_futures[future]
                try:
                    df = future.result()
                    if game_id not in results:
                        results[game_id] = {}
                    results[game_id]['news'] = df
                    logger.info(f"游戏 {game_id} 新闻爬取完成")
                except Exception as e:
                    logger.error(f"游戏 {game_id} 新闻爬取失败: {e}")
        
        return results


def main():
    """主函数示例"""
    # 创建爬虫实例，设置合理的延迟时间
    scraper = SteamScraper(delay=0.5, max_workers=2)
    
    game_id = '730'
    
    start_time = time.time()
    
    logger.info("🚀 开始爬取所有评论和新闻数据...")
    
    # 爬取所有评论数据（无数量限制）
    logger.info("📝 开始爬取所有评论...")
    reviews_df = scraper.scrape_reviews(
        game_id=game_id,
        max_reviews=60000,  # None表示爬取所有评论
        output_file=f'{game_id}_reviews_all.csv'
    )
    
    # 爬取所有新闻数据
    logger.info("📰 开始爬取所有新闻...")
    news_df = scraper.scrape_news(
        game_id=game_id,
        output_file=f'{game_id}_news_all.csv',
        max_news=None  # None表示爬取所有新闻
    )
    
    end_time = time.time()
    
    logger.info("✅ 所有数据爬取完成！")
    logger.info(f"📊 评论数量: {len(reviews_df):,}")
    logger.info(f"📰 新闻数量: {len(news_df):,}")
    logger.info(f"⏱️  总用时: {end_time - start_time:.2f} 秒")
    logger.info(f"📁 评论文件: {game_id}_reviews_all.csv")
    logger.info(f"📁 新闻文件: {game_id}_news_all.csv")


if __name__ == '__main__':
    main()