import requests
from bs4 import BeautifulSoup
import csv
import time

BASE_URL = "https://steamcommunity.com/app/413150/eventcomments/4638238612650989520?snr=2_9_100003_"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

def get_page(url):
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.text

def parse_comments(html):
    soup = BeautifulSoup(html, "html.parser")
    comments = []
    for comment_block in soup.select(".commentthread_comment "):
        user = comment_block.select_one(".commentthread_author_link").get_text(strip=True) if comment_block.select_one(".commentthread_author_link") else ""
        content = comment_block.select_one(".commentthread_comment_text").get_text(strip=True) if comment_block.select_one(".commentthread_comment_text") else ""
        time_str = comment_block.select_one(".commentthread_comment_timestamp").get_text(strip=True) if comment_block.select_one(".commentthread_comment_timestamp") else ""
        comments.append({
            "user": user,
            "content": content,
            "time": time_str
        })
    return comments

def save_to_csv(comments, filename):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "content", "time"])
        writer.writeheader()
        for c in comments:
            writer.writerow(c)

if __name__ == "__main__":
    all_comments = []
    page = 1
    while True:
        if page == 1:
            url = BASE_URL
        else:
            url = BASE_URL + f"&ctp={page}"
        print(f"正在爬取第 {page} 页: {url}")
        html = get_page(url)
        comments = parse_comments(html)
        if not comments:
            print("没有更多评论，爬取结束。")
            break
        all_comments.extend(comments)
        page += 1
        time.sleep(1) 
    save_to_csv(all_comments, "steam_event_comments.csv")
    print(f"已保存 {len(all_comments)} 条评论到 steam_event_comments.csv")
