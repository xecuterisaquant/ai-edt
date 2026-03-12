import feedparser
import time
import json
from test_engine import run_integrated_analysis # Use the integrated one we discussed

# 1. The 2026 "Alpha" Feeds
RSS_FEEDS = [
    "https://www.reuters.com/arc/outboundfeeds/rss/topics/energy/",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135",
    "https://www.hellenicshippingnews.com/category/shipping-news/tankers/feed/",
    "https://www.ogj.com/rss/general-interest"
]

# To avoid reacting to the same news twice
processed_links = set()

def start_watching():
    print("🛰️  AI EDT Watcher: Monitoring global energy & logistics...")
    print("--- Press Ctrl+C to stop ---")

    while True:
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if entry.link not in processed_links:
                        # Step 1: Filter for relevant keywords to avoid junk
                        keywords = ["oil", "venezuela", "hormuz", "tanker", "refinery", "bpd", "sanction"]
                        if any(k in entry.title.lower() for k in keywords):
                            print(f"\n[!] ALERT: {entry.title}")
                            # Step 2: Pass to the Sieve + 14B Reasoning Engine
                            run_integrated_analysis(entry.title)
                        
                        processed_links.add(entry.link)
            except Exception as e:
                print(f"⚠️ Error reading feed {url}: {e}")
        
        # Check every 2 minutes (120 seconds)
        # In a real crisis (like Hormuz closing), 120s is an eternity, but 
        # for a student setup, it prevents your CPU from pinning at 100%.
        time.sleep(120)

if __name__ == "__main__":
    start_watching()