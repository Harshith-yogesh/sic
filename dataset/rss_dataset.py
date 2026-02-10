"""Lightweight RSS dataset module for news summarization.

This module fetches RSS feeds using `feedparser` and extracts article text
with `requests` + `BeautifulSoup`. It returns a simple list of dictionaries
with keys: `title`, `category`, `source`, `published_date`, `full_text`.

The implementation is defensive: network errors and broken pages fall back
to the RSS summary or skip the article gracefully.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 8

# Mapping of canonical categories to RSS feed URLs (no API keys required)
CATEGORY_FEEDS: Dict[str, str] = {
    "politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "cinema": "https://rss.nytimes.com/services/xml/rss/nyt/Movies.xml",
    "sports": "https://www.espn.com/espn/rss/news",
    "business": "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "technology": "http://feeds.feedburner.com/TechCrunch/"
}


def extract_text_from_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Fetch a URL and try to extract the main article text.

    Strategy:
    - GET the page (timeout)
    - Parse with BeautifulSoup
    - Prefer <article> tag; otherwise concatenate <p> tags
    - Return a cleaned text string ('') if none
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "rss-dataset/1.0"})
        resp.raise_for_status()
    except Exception:
        return ""

    try:
        soup = BeautifulSoup(resp.content, "lxml")
    except Exception:
        soup = BeautifulSoup(resp.content, "html.parser")

    # Prefer <article>
    article = soup.find("article")
    paragraphs: List[str] = []
    if article:
        paragraphs = [p.get_text(strip=True) for p in article.find_all("p")]

    if not paragraphs:
        # Fallback to body paragraphs
        body = soup.find("body")
        if body:
            paragraphs = [p.get_text(strip=True) for p in body.find_all("p")]

    # Join and clean
    text = "\n\n".join([p for p in paragraphs if p and len(p) > 20])

    # If extracted text is very short, try meta description
    if len(text) < 120:
        desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if desc and desc.get("content"):
            text = desc.get("content").strip()

    return text or ""


def _parse_published(entry) -> Optional[datetime]:
    if entry is None:
        return None
    if getattr(entry, "published_parsed", None):
        try:
            return datetime.fromtimestamp(time.mktime(entry.published_parsed))
        except Exception:
            return None
    if entry.get("published"):
        try:
            # feedparser sometimes returns string
            return datetime.fromisoformat(entry.get("published"))
        except Exception:
            return None
    return None


def fetch_category_articles(category: str, max_articles: int = 5, timeout: int = DEFAULT_TIMEOUT) -> List[Dict]:
    """Fetch up to `max_articles` for the given category.

    Args:
        category: One of the supported categories (case-insensitive).
        max_articles: Max items to fetch from the feed.
        timeout: Network timeout for article fetching.

    Returns:
        List of article dicts with keys: title, category, source, published_date, full_text
    """
    cat_key = category.strip().lower()
    feed_url = CATEGORY_FEEDS.get(cat_key)
    if not feed_url:
        raise ValueError(f"Unsupported category: {category}")

    feed = feedparser.parse(feed_url)
    results: List[Dict] = []

    entries = getattr(feed, "entries", [])[:max_articles]
    for entry in entries:
        title = entry.get("title", "Untitled")
        source = feed.feed.get("title") if getattr(feed, "feed", None) else feed_url
        published_dt = _parse_published(entry)
        link = entry.get("link") or entry.get("id") or ""

        full_text = ""
        if link:
            full_text = extract_text_from_url(link, timeout=timeout)

        # Fallback to entry summary if page extraction failed
        if not full_text:
            full_text = entry.get("summary", "") or entry.get("description", "") or ""

        results.append({
            "title": title,
            "category": category,
            "source": source,
            "published_date": published_dt.isoformat() if published_dt else None,
            "full_text": full_text,
            "url": link,
        })

    return results


def fetch_all_categories(max_per_category: int = 5, timeout: int = DEFAULT_TIMEOUT) -> List[Dict]:
    """Fetch articles for all supported categories and return a flat list."""
    articles: List[Dict] = []
    for cat in CATEGORY_FEEDS.keys():
        try:
            articles.extend(fetch_category_articles(cat, max_articles=max_per_category, timeout=timeout))
        except Exception:
            # Skip broken feeds silently to keep module robust
            continue
    return articles


__all__ = ["CATEGORY_FEEDS", "fetch_category_articles", "fetch_all_categories", "extract_text_from_url"]
