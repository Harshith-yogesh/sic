"""Live RSS fetcher for real-time news retrieval.

This module exposes functions that fetch the latest articles from RSS
feeds using `feedparser`. It performs live network fetches on each call
and does not store articles on disk or in memory permanently.

Usage example:

from realtime.rss_realtime import DEFAULT_FEEDS, fetch_recent_articles

articles = fetch_recent_articles(DEFAULT_FEEDS, newspaper='BBC', category='world', max_per_feed=5)
for a in articles:
    print(a['published'], a['title'], a['link'])

Functions:
- fetch_recent_articles(feeds, newspaper=None, category=None, max_per_feed=5, max_total=100)
- get_available_newspapers(feeds)

"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional
import feedparser

# Example default feeds mapping: newspaper -> {category: feed_url}
DEFAULT_FEEDS: Dict[str, Dict[str, str]] = {
    "BBC": {
        "world": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "business": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    },
    "Reuters": {
        "world": "http://feeds.reuters.com/Reuters/worldNews",
        "business": "http://feeds.reuters.com/reuters/businessNews",
    },
    "NYTimes": {
        "world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    },
    "TheGuardian": {
        "world": "https://www.theguardian.com/world/rss",
        "business": "https://www.theguardian.com/uk/business/rss",
    },
}


def _entry_published_iso(entry) -> Optional[str]:
    """Normalize feed entry publish time to ISO 8601 string if available."""
    # feedparser provides a parsed time in `published_parsed` or `updated_parsed`.
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        # parsed is a time.struct_time
        ts = time.mktime(parsed)
        dt = datetime.fromtimestamp(ts)
        return dt.isoformat()

    # fallback to raw string if present
    raw = entry.get("published") or entry.get("updated")
    if raw:
        return raw
    return None


def fetch_recent_articles(
    feeds: Dict[str, Dict[str, str]],
    newspaper: Optional[str] = None,
    category: Optional[str] = None,
    max_per_feed: int = 5,
    max_total: int = 100,
) -> List[Dict[str, Optional[str]]]:
    """Fetch recent articles live from provided RSS feeds.

    Parameters:
    - feeds: mapping of newspaper -> {category: feed_url}
    - newspaper: optional filter to fetch only from this newspaper (case-sensitive)
    - category: optional filter to fetch only this category (case-sensitive)
    - max_per_feed: limit articles read from each feed
    - max_total: cap total returned articles

    Returns:
    - List of article dicts: {title, link, published, newspaper, category}

    Notes:
    - This function performs a network request each time it is called and
      does not persist data.
    """
    articles: List[Dict[str, Optional[str]]] = []

    for paper, cats in feeds.items():
        if newspaper and paper != newspaper:
            continue

        for cat_name, feed_url in cats.items():
            if category and cat_name != category:
                continue

            try:
                parsed = feedparser.parse(feed_url)
            except Exception:
                # network/parse error; skip this feed
                continue

            # guard against bozo feeds
            if getattr(parsed, "bozo", False):
                # still try to read entries if available
                pass

            for entry in parsed.entries[:max_per_feed]:
                title = (entry.get("title") or "").strip()
                link = entry.get("link") or ""
                published = _entry_published_iso(entry)

                articles.append(
                    {
                        "title": title,
                        "link": link,
                        "published": published,
                        "newspaper": paper,
                        "category": cat_name,
                    }
                )

            # stop early if we've reached max_total
            if len(articles) >= max_total:
                break

        if len(articles) >= max_total:
            break

    # sort by published if available (newest first); entries without published go last
    def _key(a: Dict[str, Optional[str]]):
        p = a.get("published")
        try:
            return datetime.fromisoformat(p) if p else datetime.min
        except Exception:
            # if not ISO, put it after valid dates
            return datetime.min

    articles.sort(key=_key, reverse=True)
    return articles[:max_total]


def get_available_newspapers(feeds: Dict[str, Dict[str, str]]) -> List[str]:
    """Return the list of newspaper names available in the feeds mapping."""
    return list(feeds.keys())


__all__ = [
    "DEFAULT_FEEDS",
    "fetch_recent_articles",
    "get_available_newspapers",
]
