import streamlit as st
import requests
import sys
import os
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import re

# ---------------- PATH SETUP ----------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from realtime.realtime_summarizer import RTSummarizer

st.set_page_config(page_title="📰 News Summarizer AI", layout="wide")

# ---------------- CUSTOM CSS (UI DESIGN) ----------------
st.markdown("""
<style>
.main {
    background-color: #f9fafb;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    margin-bottom: 2rem;
}

.article-card {
    background-color: white;
    padding: 1.5rem;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

.summary-box {
    background-color: #f1f5f9;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 6px solid #2563eb;
    margin-top: 1rem;
    white-space: pre-wrap;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "news_articles" not in st.session_state:
    st.session_state.news_articles = {}
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "rt_summarizer" not in st.session_state:
    st.session_state.rt_summarizer = RTSummarizer()

# ---------------- RSS FETCH ----------------
def fetch_rss(feed_url, max_articles=5):
    try:
        r = requests.get(feed_url, timeout=10)
        root = ET.fromstring(r.content)
        articles = []
        for item in root.findall(".//item")[:max_articles]:
            articles.append({
                "title": item.findtext("title", "No Title"),
                "description": item.findtext("description", ""),
                "url": item.findtext("link", ""),
                "publishedAt": item.findtext("pubDate", "")
            })
        return articles
    except Exception:
        return []

# ---------------- CATEGORY NEWS ----------------
def fetch_news(category, max_articles=5):
    rss_feeds = {
        "politics": "http://feeds.bbci.co.uk/news/politics/rss.xml",
        "technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "sports": "http://feeds.bbci.co.uk/sport/rss.xml",
        "health": "http://feeds.bbci.co.uk/news/health/rss.xml",
        "finance": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "cinema": "https://www.hollywoodreporter.com/t/movies/feed/"
    }
    return fetch_rss(rss_feeds.get(category), max_articles)

# ---------------- ARTICLE SCRAPER ----------------
def fetch_full_article(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
        return " ".join(p for p in paragraphs if len(p) > 60)
    except Exception:
        return ""

# ---------------- RAW SUMMARIZER ----------------
def summarize(text):
    return st.session_state.rt_summarizer.summarize(
        text,
        num_sentences=30,
        use_position_weighting=True
    )

# ---------------- SMART FORMATTER ----------------
def format_summary(text):
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cleaned = []
    for s in sentences:
        if len(s.split()) > 30:
            parts = re.split(r',|;| and ', s)
            cleaned.extend([p.strip().capitalize() for p in parts if len(p.split()) > 6])
        else:
            cleaned.append(s.strip().capitalize())

    overview = " ".join(cleaned[:3])
    key_points = cleaned[3:10]

    health_keywords = [
        "death", "cancer", "heart", "brain", "babies", "womb",
        "inflammation", "disease", "health"
    ]
    health_risks = [s for s in cleaned if any(k in s.lower() for k in health_keywords)]

    impact = " ".join(cleaned[10:14])

    prevention = [
        s for s in cleaned if any(
            k in s.lower() for k in ["avoid", "walk", "mask", "reduce", "minimise"]
        )
    ]

    result = "📌 OVERVIEW\n" + overview + "\n\n"
    result += "📍 KEY POINTS\n"
    for p in key_points[:5]:
        result += f"• {p}\n"

    if health_risks:
        result += "\n🚨 HEALTH RISKS\n"
        for h in health_risks[:5]:
            result += f"• {h}\n"

    result += "\n📊 WHY THIS MATTERS\n" + impact + "\n\n"

    if prevention:
        result += "🧾 WHAT YOU CAN DO\n"
        for p in prevention[:4]:
            result += f"• {p}\n"

    return result

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🧭 Control Panel")
    st.markdown("Select categories and fetch **live news updates**.")

    categories = {
        "🏛️ Politics": "politics",
        "🎬 Cinema": "cinema",
        "⚽ Sports": "sports",
        "💻 Technology": "technology",
        "💰 Finance": "finance",
        "🏥 Health": "health"
    }

    selected = st.multiselect(
        "Select Categories",
        categories.keys(),
        default=["🏛️ Politics", "💻 Technology"]
    )

    if st.button("🔄 Fetch Live News"):
        st.session_state.news_articles = {}
        st.session_state.summaries = {}
        for c in selected:
            st.session_state.news_articles[c] = fetch_news(categories[c], 3)
        st.success("Live news loaded!")

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>📰 Real-Time News Summarizer AI</h1>
    <p>
        Real-time news → Structured summaries → Health alerts → Downloadable reports
    </p>
    <p>⚡ Fast • 📄 One-Page • 📥 Download</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN CONTENT ----------------
if not st.session_state.news_articles:
    st.info("👈 Use the Control Panel to fetch live news")
else:
    for category, articles in st.session_state.news_articles.items():
        st.subheader(category)

        for i, article in enumerate(articles):
            key = f"{category}_{i}"

            st.markdown('<div class="article-card">', unsafe_allow_html=True)
            st.markdown(f"### 📰 {article['title']}")
            st.markdown(f"📅 **Published:** {article['publishedAt']}")
            st.markdown(f"📝 {article['description']}")

            if st.button("✨ Generate Structured Summary", key=key):
                with st.spinner("Analyzing article..."):
                    full_text = fetch_full_article(article["url"])
                    raw = summarize(full_text)
                    st.session_state.summaries[key] = format_summary(raw)
                    st.rerun()

            if key in st.session_state.summaries:
                st.markdown(
                    f"<div class='summary-box'>{st.session_state.summaries[key]}</div>",
                    unsafe_allow_html=True
                )

                st.download_button(
                    "📥 Download Summary",
                    st.session_state.summaries[key],
                    file_name="news_summary.txt",
                    mime="text/plain"
                )

            st.markdown("</div>", unsafe_allow_html=True)
