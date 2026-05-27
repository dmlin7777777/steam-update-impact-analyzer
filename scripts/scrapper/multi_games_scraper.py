from scrapper_optimized import SteamScraper
import time
from datetime import datetime
import os
def scrape_multiple_games_example():
    """使用多线程并发爬取多个游戏"""
    
    # 定义要爬取的游戏列表
    game_ids = [
        
        '730',
        '1938090'
       
            # 您的目标游戏
        # 添加更多游戏ID...
    ]
    
    print("🎮 多游戏并发爬取工具")
    print("=" * 50)
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 目标游戏数量: {len(game_ids)}")
    print(f"🎯 游戏列表: {', '.join(game_ids)}")
    
    # 创建爬虫实例 - 使用多线程
    scraper = SteamScraper(delay=0.5, max_workers=3)  # 3个线程并发
    
    start_time = time.time()
    
    if not os.path.exists("./multi_games_data"):
        os.makedirs("./multi_games_data")

    # 使用多线程并发爬取多个游戏
    results = scraper.scrape_multiple_games(
        game_ids=game_ids,
        base_output_dir="./multi_games_data"  # 输出目录
    )
    
    end_time = time.time()
    
    # 统计结果
    total_reviews = 0
    total_news = 0
    success_count = 0
    
    print("\n" + "=" * 60)
    print("📊 爬取结果统计")
    print("=" * 60)
    
    for game_id, data in results.items():
        if 'reviews' in data and 'news' in data:
            reviews_count = len(data['reviews'])
            news_count = len(data['news'])
            total_reviews += reviews_count
            total_news += news_count
            success_count += 1
            print(f"✅ 游戏 {game_id}: 评论 {reviews_count:,} 条, 新闻 {news_count:,} 条")
        else:
            print(f"❌ 游戏 {game_id}: 爬取失败")
    
    print(f"\n📈 总体统计:")
    print(f"   - 成功游戏: {success_count}/{len(game_ids)}")
    print(f"   - 总评论数: {total_reviews:,}")
    print(f"   - 总新闻数: {total_news:,}")
    print(f"   - 总用时: {end_time - start_time:.2f} 秒")
    print(f"   - 平均每游戏: {(end_time - start_time)/len(game_ids):.1f} 秒")
    print(f"📁 输出目录: ./multi_games_data/")

if __name__ == "__main__":
    scrape_multiple_games_example()