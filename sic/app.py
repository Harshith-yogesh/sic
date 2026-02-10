import streamlit as st
import requests
from datetime import datetime, timedelta
import re
from collections import Counter
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from realtime.realtime_summarizer import RTSummarizer

st.set_page_config(page_title="📰 News Summarizer AI (NLP)", layout="wide", initial_sidebar_state="expanded")
st.set_option("client.showErrorDetails", True)

# Initialize session state
if "news_articles" not in st.session_state:
    st.session_state.news_articles = {}  # Dict for categories
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "rt_summarizer" not in st.session_state:
    st.session_state.rt_summarizer = RTSummarizer()  # Real-time LSA summarizer

# Function to generate summary using NLP
def generate_nlp_summary(article_text, model_type="LSA (Recommended)", sentences_count=3, preprocess_options=None, extra_stopwords=None):
    """Generate a concise summary using pure Python NLP algorithms.

    Args:
        article_text (str): Full article text.
        model_type (str): "LSA (Recommended)" or "Luhn (Alternative)".
        sentences_count (int): Number of sentences to return.
        preprocess_options (dict): {'remove_punct': bool, 'remove_numbers': bool, 'lowercase': bool}
        extra_stopwords (set|None): Additional stopwords to exclude.
    """
    if not article_text or article_text.strip() == "":
        return "❌ Error: No article text provided."

    try:
        # Default preprocessing options
        if preprocess_options is None:
            preprocess_options = {"remove_punct": True, "remove_numbers": True, "lowercase": True}

        text = article_text
        if preprocess_options.get("lowercase", False):
            text = text.lower()
        if preprocess_options.get("remove_punct", False):
            text = re.sub(r"[^\w\s]", " ", text)
        if preprocess_options.get("remove_numbers", False):
            text = re.sub(r"\d+", " ", text)

        # Split into sentences (use original text for sentence boundaries when possible)
        sentences = re.split(r'(?<=[.!?])\s+', article_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= sentences_count:
            return " ".join(sentences)

        # Tokenize words and clean (on preprocessed text)
        words = re.findall(r"\b\w+\b", text)

        # Default stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'it', 'its', 'their', 'they', 'he', 'she', 'we', 'you', 'i', 'our'
        }

        if extra_stopwords:
            stop_words = stop_words.union(set([w.strip().lower() for w in extra_stopwords if w.strip()]))

        important_words = [w for w in words if w not in stop_words and len(w) > 2]
        word_freq = Counter(important_words)

        if model_type == "LSA (Recommended)":
            # Score sentences by word frequency (LSA-inspired approach)
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                score = 0
                # Preprocess sentence similarly for scoring
                s_text = sentence.lower() if preprocess_options.get("lowercase", False) else sentence
                if preprocess_options.get("remove_punct", False):
                    s_text = re.sub(r"[^\w\s]", " ", s_text)
                if preprocess_options.get("remove_numbers", False):
                    s_text = re.sub(r"\d+", " ", s_text)

                sentence_words = re.findall(r"\b\w+\b", s_text)
                for word in sentence_words:
                    if word in word_freq:
                        score += word_freq[word]
                sentence_scores[i] = score / (len(sentence_words) + 1) if sentence_words else 0
        else:
            # Luhn: Based on word frequency and position
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                score = 0
                s_text = sentence.lower() if preprocess_options.get("lowercase", False) else sentence
                if preprocess_options.get("remove_punct", False):
                    s_text = re.sub(r"[^\w\s]", " ", s_text)
                if preprocess_options.get("remove_numbers", False):
                    s_text = re.sub(r"\d+", " ", s_text)

                sentence_words = re.findall(r"\b\w+\b", s_text)
                for word in sentence_words:
                    if word in word_freq:
                        score += word_freq[word]
                # Give slight bonus to earlier sentences
                position_bonus = 1.0 + (1.0 - (i / len(sentences)))
                sentence_scores[i] = (score / (len(sentence_words) + 1)) * position_bonus if sentence_words else 0

        # Get top sentences by score
        top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:sentences_count]
        top_indices = sorted([idx for idx, _ in top_indices])  # Sort by original order

        summary_sentences = [sentences[i] for i in top_indices]
        summary_text = " ".join(summary_sentences)

        return summary_text if summary_text else "⚠️ Could not generate summary from this article."
    except Exception as e:
        return f"❌ Summarization error: {str(e)}"

# Real-time summarizer function (optimized for live articles)
def summarize_with_realtime(article_text, num_sentences=3, use_position_weighting=True):
    """Use RTSummarizer for fast real-time article summarization.
    
    Args:
        article_text (str): Raw article text
        num_sentences (int): Number of sentences in summary
        use_position_weighting (bool): Boost early sentences
    
    Returns:
        str: Extracted summary
    """
    try:
        return st.session_state.rt_summarizer.summarize(
            article_text,
            num_sentences=num_sentences,
            use_position_weighting=use_position_weighting
        )
    except Exception as e:
        return f"⚠️ Summarization error: {str(e)}"

# Function to fetch news from NewsAPI
def fetch_news(query="technology", language="en", max_articles=5):
    """Fetch news articles from NewsAPI."""
    try:
        # Using newsapi.org free tier - no key needed for basic search
        url = f"https://newsapi.org/v2/everything?q={query}&language={language}&sortBy=publishedAt&pageSize={max_articles}"
        
        # Alternative: Use a free news source (BBC, Reuters, etc.)
        # For demo, we'll use a hardcoded set if API fails
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('articles'):
                return data['articles']
        
        # Fallback: Return demo articles
        return get_demo_articles(query)
    except Exception as e:
        st.warning(f"Could not fetch live news: {str(e)}. Using demo articles.")
        return get_demo_articles(query)


def fetch_rss(feed_url, max_articles=5):
    """Fetch simple RSS feed items using built-in XML parsing."""
    try:
        resp = requests.get(feed_url, timeout=10)
        if resp.status_code != 200:
            return []
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        items = []
        # Support common RSS structure
        for item in root.findall('.//item')[:max_articles]:
            title = item.findtext('title') or 'No Title'
            desc = item.findtext('description') or ''
            pub = item.findtext('pubDate') or ''
            link = item.findtext('link') or ''
            items.append({
                'title': title,
                'description': desc,
                'content': desc,
                'source': {'name': feed_url},
                'publishedAt': pub,
                'url': link,
                'urlToImage': ''
            })
        return items
    except Exception:
        return []

# Demo articles for testing
def get_demo_articles(query):
    """Return demo articles for testing with categorized content."""
    
    all_demo_articles = {
        "politics": [
            {
                "title": "Government Announces Major Economic Policy Reform for 2026",
                "description": "New fiscal policies aimed at strengthening the economy and supporting middle-class families.",
                "content": "The government has unveiled a comprehensive economic reform package designed to boost economic growth and improve living standards. The plan includes tax incentives for small businesses, infrastructure investments worth billions, and support for green energy initiatives. Economists predict these measures could increase GDP growth by 2-3% over the next two years. The policy also addresses healthcare and education funding. Congress is expected to vote on these measures next month with bipartisan support anticipated.",
                "source": {"name": "Political Times"},
                "publishedAt": datetime.now().isoformat(),
                "url": "https://example.com/politics-1",
                "urlToImage": "https://via.placeholder.com/400x300?text=Politics"
            },
            {
                "title": "International Summit Discusses Climate Change and Trade",
                "description": "World leaders meet to address global challenges and strengthen diplomatic relations.",
                "content": "Delegates from over 150 countries gathered for the annual International Summit to discuss pressing global issues. Key topics included climate change mitigation, trade agreements, and cybersecurity threats. The summit produced several bilateral agreements aimed at reducing carbon emissions and promoting sustainable development. Leaders emphasized the importance of international cooperation in facing shared challenges.",
                "source": {"name": "Global News Network"},
                "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                "url": "https://example.com/politics-2",
                "urlToImage": "https://via.placeholder.com/400x300?text=Global+Politics"
            },
        ],
        "cinema": [
            {
                "title": "Oscar Nominations Announced: Record-Breaking Year for Independent Films",
                "description": "This year's Academy Award nominations celebrate diverse storytelling and outstanding performances.",
                "content": "The Academy announced its nominations for the upcoming Oscars, with several surprises that reflect changing tastes in cinema. Independent films received an unprecedented number of nominations across major categories. Several international films made the Best Picture shortlist, signaling growing recognition of global cinema. Major streaming platforms also received significant nominations. Industry insiders predict a competitive year with no clear frontrunner in the top categories.",
                "source": {"name": "Cinema Today"},
                "publishedAt": datetime.now().isoformat(),
                "url": "https://example.com/cinema-1",
                "urlToImage": "https://via.placeholder.com/400x300?text=Cinema"
            },
            {
                "title": "Blockbuster Action Film Breaks Box Office Records",
                "description": "Latest superhero movie sets new records on its opening weekend worldwide.",
                "content": "A highly anticipated action film has shattered box office records, earning over 500 million dollars in its opening weekend globally. The film features groundbreaking visual effects and an ensemble cast of A-list actors. Critics praised the film's storytelling and character development alongside its spectacular action sequences. The success has already led studios to announce sequels and spin-offs.",
                "source": {"name": "Entertainment Weekly"},
                "publishedAt": (datetime.now() - timedelta(days=2)).isoformat(),
                "url": "https://example.com/cinema-2",
                "urlToImage": "https://via.placeholder.com/400x300?text=Blockbuster"
            },
        ],
        "industry": [
            {
                "title": "Manufacturing Sector Shows Strong Growth as Companies Invest in Automation",
                "description": "Industrial output increases as businesses modernize production facilities with AI and robotics.",
                "content": "The manufacturing sector reported robust growth this quarter, driven by increased investment in automation and advanced manufacturing technologies. Companies are implementing AI-powered quality control systems and robotic production lines to improve efficiency and reduce costs. Supply chain optimization has also contributed to improved productivity. Industry experts expect this growth trend to continue as more companies embrace digital transformation.",
                "source": {"name": "Industry Insights"},
                "publishedAt": datetime.now().isoformat(),
                "url": "https://example.com/industry-1",
                "urlToImage": "https://via.placeholder.com/400x300?text=Industry"
            },
            {
                "title": "Major Tech Companies Announce Merger Creating Industry Giant",
                "description": "Two leading technology firms merge to create the world's largest software company.",
                "content": "In a deal valued at over 100 billion dollars, two major technology companies have announced a merger that will create an industry powerhouse. The combined entity will have a workforce of over 500,000 employees across multiple continents. The merger is expected to accelerate innovation in cloud computing and artificial intelligence. Regulatory authorities are reviewing the deal, with approval expected within six months.",
                "source": {"name": "Business Daily"},
                "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                "url": "https://example.com/industry-2",
                "urlToImage": "https://via.placeholder.com/400x300?text=Tech+Merger"
            },
        ],
        "sports": [
            {
                "title": "Championship Team Wins Historic Victory in Thrilling Finals Match",
                "description": "Underdog team captures title after dramatic overtime performance.",
                "content": "In a stunning upset, an underdog team defeated the defending champions in the finals, capturing their first championship title in team history. The final match went into multiple overtimes with both teams displaying exceptional skill and determination. The winning goal came with just 30 seconds remaining in the final overtime period. The victory sparked celebrations across the city with thousands of fans gathering for a parade.",
                "source": {"name": "Sports Central"},
                "publishedAt": datetime.now().isoformat(),
                "url": "https://example.com/sports-1",
                "urlToImage": "https://via.placeholder.com/400x300?text=Sports"
            },
            {
                "title": "Athlete Breaks World Record in Track and Field Event",
                "description": "Olympic champion sets new international record that stood for 25 years.",
                "content": "A renowned athlete has broken a world record that had stood for over two decades, finishing the 1500-meter race in an unprecedented time. The performance took place at an international athletics championship with tens of thousands of spectators watching. The athlete credits years of dedicated training and new training methodologies for the breakthrough performance. This achievement solidifies their position as one of the greatest athletes of all time.",
                "source": {"name": "Athletic Times"},
                "publishedAt": (datetime.now() - timedelta(days=2)).isoformat(),
                "url": "https://example.com/sports-2",
                "urlToImage": "https://via.placeholder.com/400x300?text=World+Record"
            },
        ],
        "agriculture": [
            {
                "title": "Sustainable Farming Methods Show Increased Crop Yields",
                "description": "Farmers using eco-friendly practices report higher productivity and better profitability.",
                "content": "Agricultural research shows that sustainable and organic farming methods can achieve yields comparable to conventional farming while reducing environmental impact. Farmers implementing crop rotation, composting, and integrated pest management report better soil health and increased productivity. These methods also reduce chemical usage and lower production costs over time. Government initiatives are encouraging more farmers to adopt sustainable practices through subsidies and educational programs.",
                "source": {"name": "Farming Today"},
                "publishedAt": datetime.now().isoformat(),
                "url": "https://example.com/agriculture-1",
                "urlToImage": "https://via.placeholder.com/400x300?text=Agriculture"
            },
            {
                "title": "Breakthrough in Vertical Farming Technology Revolutionizes Urban Agriculture",
                "description": "New indoor farming systems allow cities to produce fresh food locally and sustainably.",
                "content": "Innovative vertical farming systems are transforming urban agriculture by allowing fresh produce to be grown in high-rise facilities using minimal water and no pesticides. These systems utilize LED lighting and hydroponic technology to create optimal growing conditions year-round. Cities are adopting vertical farms to improve food security and reduce transportation emissions. Initial results show these systems can be economically viable while providing fresh, local produce to urban populations.",
                "source": {"name": "Green Agriculture Review"},
                "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                "url": "https://example.com/agriculture-2",
                "urlToImage": "https://via.placeholder.com/400x300?text=Vertical+Farming"
            },
        ],
        "technology": [
            {
                "title": "AI and Machine Learning Advances in 2026",
                "description": "Latest breakthroughs in artificial intelligence including NLP models, computer vision, and autonomous systems.",
                "content": "Artificial intelligence continues to advance rapidly with groundbreaking developments. Natural Language Processing models have reached new heights, enabling machines to understand and generate human-like text with unprecedented accuracy. Recent breakthroughs in transformer-based architectures have revolutionized how we process language. Computer vision applications are becoming more sophisticated, allowing for better image recognition and analysis. Autonomous systems are becoming safer and more reliable. Companies are investing billions into AI research and development.",
                "source": {"name": "Tech News Daily"},
                "publishedAt": datetime.now().isoformat(),
                "url": "https://example.com/tech-1",
                "urlToImage": "https://via.placeholder.com/400x300?text=AI"
            },
            {
                "title": "Quantum Computing Reaches New Milestone in Error Correction",
                "description": "Scientists achieve breakthrough in quantum error correction enabling more stable quantum computers.",
                "content": "Quantum computing researchers have achieved a major milestone in quantum error correction, bringing practical quantum computers closer to reality. The breakthrough reduces errors in quantum computations by an order of magnitude, allowing for longer and more complex computations. Multiple research institutions have contributed to this advancement through collaborative efforts. This progress suggests commercial quantum computers could become available within the next 3-5 years.",
                "source": {"name": "Innovation Weekly"},
                "publishedAt": (datetime.now() - timedelta(days=2)).isoformat(),
                "url": "https://example.com/tech-2",
                "urlToImage": "https://via.placeholder.com/400x300?text=Quantum"
            },
        ],
        "finance": [
            {
                "title": "Stock Market Reaches All-Time High Amid Economic Growth",
                "description": "Major indices climb to record levels as investors show confidence in economic outlook.",
                "content": "Global stock markets have reached new all-time highs, driven by strong corporate earnings and positive economic indicators. Central banks have signaled that interest rate increases may be complete, giving investors confidence about future economic conditions. Technology and renewable energy sectors have led the gains. Analysts expect continued market strength if inflation continues to decline and corporate profits remain robust.",
                "source": {"name": "Financial Times"},
                "publishedAt": datetime.now().isoformat(),
                "url": "https://example.com/finance-1",
                "urlToImage": "https://via.placeholder.com/400x300?text=Stock+Market"
            },
            {
                "title": "Cryptocurrency Market Gains Mainstream Adoption from Financial Institutions",
                "description": "Traditional banks and investment firms increase cryptocurrency holdings and services.",
                "content": "Major financial institutions are increasingly integrating cryptocurrency services and holdings into their operations. Several large banks have launched cryptocurrency trading desks and custody services. This institutional adoption is seen as a turning point for cryptocurrency market legitimacy and stability. Regulatory clarity has also improved with governments establishing frameworks for cryptocurrency oversight. The market capitalization of cryptocurrencies continues to grow as institutional money flows in.",
                "source": {"name": "Crypto News Hub"},
                "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                "url": "https://example.com/finance-2",
                "urlToImage": "https://via.placeholder.com/400x300?text=Cryptocurrency"
            },
        ],
        "health": [
            {
                "title": "New Treatment Shows Promise in Fighting Chronic Diseases",
                "description": "Breakthrough therapy offers hope for millions suffering from previously incurable conditions.",
                "content": "Medical researchers have announced a promising new treatment that shows effectiveness against several chronic diseases. The therapy uses advanced genetic engineering techniques to modify immune cells to fight disease. Clinical trials show remission rates exceeding 70% in patients with certain conditions previously considered incurable. The treatment is expected to enter wider clinical trials this year with potential regulatory approval within 2-3 years.",
                "source": {"name": "Medical Today"},
                "publishedAt": datetime.now().isoformat(),
                "url": "https://example.com/health-1",
                "urlToImage": "https://via.placeholder.com/400x300?text=Medicine"
            },
            {
                "title": "Mental Health Support Programs Expand in Schools and Workplaces",
                "description": "Organizations recognize importance of mental health and increase counseling and support services.",
                "content": "Educational institutions and corporations are significantly expanding mental health support programs, recognizing the importance of psychological well-being. Schools are adding more counselors and implementing mental health awareness programs. Workplaces are offering mental health days, counseling services, and stress management programs. Studies show that these initiatives improve academic performance, productivity, and overall well-being. Mental health advocacy groups applaud these developments as steps toward destigmatizing mental health issues.",
                "source": {"name": "Health Weekly"},
                "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                "url": "https://example.com/health-2",
                "urlToImage": "https://via.placeholder.com/400x300?text=Health"
            },
        ],
    }
    
    # Match query to category
    query_lower = query.lower()
    
    # Check which category the query matches
    for category, articles in all_demo_articles.items():
        if category in query_lower or any(word in query_lower for word in category.split()):
            return articles
    
    # Return default tech articles if no match
    return all_demo_articles.get("technology", [])

# Sidebar setup
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Category selection
    st.subheader('📑 News Categories')
    categories = {
        "🏛️ Politics": "politics",
        "🎬 Cinema": "cinema OR movies OR entertainment",
        "🏭 Industry": "industry OR manufacturing",
        "💼 Business": "business OR corporate OR startup",
        "⚽ Sports": "sports",
        "🌾 Agriculture": "agriculture OR farming",
        "🌍 World": "world OR international OR global",
        "💻 Technology": "technology OR AI",
        "💰 Finance": "finance OR stock market OR economy",
        "🏥 Health": "health OR medicine OR healthcare"
    }
    
    selected_categories = st.multiselect(
        "Select news categories:",
        options=list(categories.keys()),
        default=["🏛️ Politics", "🎬 Cinema", "⚽ Sports"],
        help="Select multiple categories to view news"
    )
    
    # News source selection
    st.subheader('📡 News Source (optional)')
    source_options = [
        "NewsAPI (default)",
        "BBC RSS",
        "CNN RSS",
        "NDTV RSS",
        "ESPN RSS"
    ]
    news_source = st.selectbox("Choose news source:", source_options, index=0, help="Choose a source or use default NewsAPI-style fetch")

    # NLP settings
    st.subheader('🧠 NLP Summarization Algorithm')
    model_choice = st.selectbox(
        "Choose summarization algorithm:",
        ["LSA (Recommended)", "Luhn (Alternative)", "⚡ Real-Time LSA (Fast)"],
        help="LSA: Better quality. Luhn: Faster. Real-Time: Optimized for live articles."
    )
    
    # Real-time summarizer options
    if model_choice == "⚡ Real-Time LSA (Fast)":
        st.info("✨ Using optimized real-time LSA summarizer for instant processing")
        use_position_weighting = st.checkbox("Boost early sentences", value=True, help="Give higher priority to sentences at the beginning of the article")
    else:
        use_position_weighting = False

    # Summary length
    st.subheader('📝 Summary Length')
    sentences_count = st.slider("Sentences per summary:", 1, 5, 2, help="Number of sentences in each summary")

    # Preprocessing options
    st.subheader('⚙️ Preprocessing (affects NLP)')
    remove_punct = st.checkbox("Remove punctuation", value=True)
    remove_numbers = st.checkbox("Remove numbers", value=True)
    lowercase = st.checkbox("Lowercasing", value=True)
    preprocess_options = {"remove_punct": remove_punct, "remove_numbers": remove_numbers, "lowercase": lowercase}

    # Stopword handling
    st.subheader('🔤 Stopword Handling')
    use_default_stopwords = st.checkbox("Use default NLP stopwords", value=True)
    extra_stopwords_input = st.text_input("Extra stopwords (comma-separated)", value="", help="Add additional stopwords to ignore (comma separated)")
    custom_stopwords = [w.strip() for w in extra_stopwords_input.split(',')] if extra_stopwords_input.strip() else []

    st.info(f"""
    **{model_choice}**

    ✅ Pure NLP-Based (No API keys)
    ✅ Runs on your computer
    ✅ Completely FREE
    """)
    
    # Fetch news button
    if st.button("🔄 Fetch Categorized News", use_container_width=True):
        with st.spinner("📡 Fetching news from all categories..."):
            st.session_state.news_articles = {}
            # Map simple RSS feed names
            rss_map = {
                "BBC RSS": "http://feeds.bbci.co.uk/news/rss.xml",
                "CNN RSS": "http://rss.cnn.com/rss/edition.rss",
                "NDTV RSS": "https://feeds.feedburner.com/NDTV-News",
                "ESPN RSS": "https://www.espn.com/espn/rss/news"
            }

            for category_display, category_query in categories.items():
                if category_display in selected_categories:
                    if news_source == "NewsAPI (default)":
                        articles = fetch_news(category_query, max_articles=3)
                    else:
                        feed_url = rss_map.get(news_source)
                        articles = fetch_rss(feed_url, max_articles=3) if feed_url else fetch_news(category_query, max_articles=3)
                    st.session_state.news_articles[category_display] = articles
            st.success(f"✅ News loaded for {len(selected_categories)} categories!")
    
    # Info
    st.divider()
    st.subheader("📚 About This App")
    st.markdown("""
    **Categorized News Summarizer**
    
    - Fetch news by category
    - Pure NLP summarization
    - No subscriptions needed
    - Real-time news feeds
    """)

# Main content
st.title("📰 Categorized News Summarizer")
st.markdown("**Organized by Category with AI-Powered Summaries**")

# Display news by category
if not st.session_state.news_articles:
    st.info("👈 Select categories and click 'Fetch Categorized News' in the sidebar to get started!")
else:
    for category, articles in st.session_state.news_articles.items():
        if articles:
            st.markdown(f"## {category}")
            
            for idx, article in enumerate(articles):
                unique_key = f"{category}_{idx}"
                
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.subheader(article.get('title', 'No Title')[:80])
                        source = article.get('source', {}).get('name', 'Unknown')
                        date = article.get('publishedAt', 'Unknown')[:10]
                        st.caption(f"📌 {source} | 📅 {date}")
                        st.write(article.get('description', 'No description'))
                        
                        article_full_text = article.get('content', '') or article.get('description', '')
                    
                    with col2:
                        if st.button("✨ Summarize", key=f"btn_{unique_key}", use_container_width=True):
                            with st.spinner(f"🤖 Processing..."):
                                # Use real-time summarizer if selected
                                if model_choice == "⚡ Real-Time LSA (Fast)":
                                    summary = summarize_with_realtime(
                                        article_full_text,
                                        num_sentences=sentences_count,
                                        use_position_weighting=use_position_weighting
                                    )
                                else:
                                    # Fall back to traditional NLP summarizer
                                    extra_sw = custom_stopwords if use_default_stopwords is False else custom_stopwords
                                    summary = generate_nlp_summary(article_full_text, model_choice, sentences_count, preprocess_options=preprocess_options, extra_stopwords=extra_sw)
                                st.session_state.summaries[unique_key] = summary
                                st.rerun()
                    
                    # Display summary
                    if unique_key in st.session_state.summaries:
                        st.divider()
                        summary = st.session_state.summaries[unique_key]
                        
                        if summary.startswith("❌") or summary.startswith("⚠️"):
                            st.warning(summary)
                        elif summary.startswith("ℹ️"):
                            st.info(summary)
                        else:
                            st.success("✅ Summary:")
                            st.info(summary)
                        
                        col_down, col_clear = st.columns(2)
                        with col_down:
                            st.download_button(
                                "📥 Download",
                                data=summary,
                                file_name=f"summary_{unique_key}.txt",
                                key=f"dl_{unique_key}",
                                use_container_width=True
                            )
                        with col_clear:
                            if st.button("🗑️ Clear", key=f"clear_{unique_key}", use_container_width=True):
                                del st.session_state.summaries[unique_key]
                                st.rerun()
            
            st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>📰 Categorized News Summarizer</strong> | Pure NLP-Based | No API Keys Required</p>
    <p style='font-size: 12px; color: gray;'>Categories: Politics • Cinema • Industry • Business • World • Sports • Agriculture • Technology • Finance • Health</p>
</div>
""", unsafe_allow_html=True)

