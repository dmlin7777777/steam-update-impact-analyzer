import requests
import pandas as pd
import time

headers = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/85.0.4183.102 Safari/537.36', 'Accept-Language': 'zh-CN '
}
def review(game_id, sleep_time=0.5):
    url = f'https://store.steampowered.com/appreviews/{game_id}?json=1'
    params = {'filter': 'recent', 'language': 'all', 'cursor': '*', 'purchase_type': 'all', 'num_per_page': 100}
    url_json = requests.get(url, params=params,headers=headers)
    review_data = []
    data = url_json.json()
    while data['reviews']: 
        current_cursor = data['cursor']
        for review in data['reviews']:
            review_entry = {
                'SteamID': review['author']['steamid'],
                'played_hours': review['author']['playtime_forever'],
                'num_games_owned': review['author']['num_games_owned'] if review['author'].get('num_games_owned') != 0 else 'Private Account',
                'num_reviews': review['author']['num_reviews'],
                'playtime_last_two_weeks': review['author']['playtime_last_two_weeks'],
                'playtime_at_review': review['author']['playtime_at_review'],
                'last_played': time.strftime("%Y-%m-%d", time.gmtime(review['author']['last_played'])) if review['author'].get('last_played') else '',
                'language': review['language'],
                'review_content': review['review'],
                'voted_up': review['voted_up'],
                'review_date': time.strftime("%Y-%m-%d",time.gmtime(review['timestamp_created'] if review.get('timestamp_created') else 0)),
                'last_updated': time.strftime("%Y-%m-%d", time.gmtime(review.get('timestamp_updated', 0))),
                'votes_up': review['votes_up'],
                'votes_funny': review['votes_funny'],
                'weighted_vote_score': review['weighted_vote_score'],
                'comment_count': review['comment_count'],
                'steam_purchase': review['steam_purchase'],
                'received_for_free': review['received_for_free'],
                'written_during_early_access': review['written_during_early_access'],
                'primarily_steam_deck': review['primarily_steam_deck']
            }
            review_data.append(review_entry)
        print(f"Fetched {len(data['reviews'])} reviews, total so far: {len(review_data)}")
        page = (len(review_data) // params['num_per_page']) + 1
        print(f"Moving to page {page} with cursor {current_cursor}")
        params['cursor'] = current_cursor
        url_json = requests.get(url, params=params, headers=headers)
        data = url_json.json()
        time.sleep(sleep_time)
        if current_cursor == data['cursor']:
                print("Cursor did not change, stopping to avoid infinite loop.")
                break
    df = pd.DataFrame(review_data)
    df[['voted_up','steam_purchase','received_for_free','written_during_early_access','primarily_steam_deck']] = df[['voted_up','steam_purchase','received_for_free','written_during_early_access','primarily_steam_deck']].map(lambda x: 1 if x else 0)
    df.to_csv(r'C:/Users/12932\Desktop/nus/BAP/2668510.csv', index=False ,encoding='utf-8-sig')
    print("The data has been successfully saved to the table")

def news(game_id):
    url = f'https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid={game_id}&count=1'
    url_rss = requests.get(url, headers=headers)
    data = url_rss.json()
    count = data['appnews']['count']
    url_rss = requests.get(f'https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid={game_id}&count={count}', headers=headers)
    news_data = []
    data = url_rss.json()
    for news in data['appnews']['newsitems']:
        news_entry = {
            'gid': news['gid'],
            'title': news['title'],
            'url': news['url'],
            'is_external_url': news['is_external_url'],
            'author': news['author'],
            'contents': news['contents'],
            'feedlabel': news['feedlabel'],
            'date': time.strftime("%Y-%m-%d", time.gmtime(news['date'])),
            'feedname': news['feedname'],
            'appid': news['appid']
        }
        news_data.append(news_entry)
    df = pd.DataFrame(news_data)
    df.to_csv(r'C:/Users/12932\Desktop/nus/BAP/2668510_news.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    game_id = '730'
    start_time = time.time()
    review(game_id=game_id, sleep_time=0.5)
    news(game_id=game_id)
    end_time = time.time()
    print(f"\n运行结束共用时{float(end_time-start_time)}秒")