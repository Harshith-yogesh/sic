import streamlit as st
import requests
import sys
import os
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import re

# ---------------- PATH SETUP ----------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.dirname(os.path.abspath(__file__))  # d:/Sic/sic
sys.path.insert(0, parent_dir)
from realtime.realtime_summarizer import RTSummarizer

# Try to import neural classifier
try:
    from classifier.neural_classifier import NeuralNewsClassifier
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

st.set_page_config(page_title="📰 BBC News Summarizer & Classifier", layout="wide")

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
    st.session_state.news_articles = []  # List of raw articles (not pre-categorized)
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "rt_summarizer" not in st.session_state:
    st.session_state.rt_summarizer = RTSummarizer()
if "classifier" not in st.session_state:
    st.session_state.classifier = None
if "classifications" not in st.session_state:
    st.session_state.classifications = {}  # key -> (category, confidence)
if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []  # User-selected categories to display

# ---------------- CLASSIFIER LOADING ----------------
def load_classifier():
    """Load the trained neural network classifier if available."""
    if NEURAL_AVAILABLE:
        neural_path = os.path.join(script_dir, "models", "neural_classifier.pt")
        if os.path.exists(neural_path):
            try:
                classifier = NeuralNewsClassifier()
                classifier.load_model(neural_path)
                return classifier
            except Exception:
                return None
    return None

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

# ---------------- BBC NEWS FEEDS ----------------
# General BBC feeds - model does the classification, NOT the feeds
BBC_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.bbci.co.uk/news/uk/rss.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "http://feeds.bbci.co.uk/sport/rss.xml",
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
]

def fetch_all_bbc_news():
    """Fetch from ALL BBC feeds - let the AI model classify them."""
    all_articles = []
    seen_titles = set()
    
    for feed_url in BBC_FEEDS:
        articles = fetch_rss(feed_url, max_articles=30)
        for article in articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                all_articles.append(article)
    
    return all_articles

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
    st.header("⚙️ Settings")
    
    num_articles = st.slider(
        "Articles per Category",
        min_value=3,
        max_value=15,
        value=5,
        help="How many news summaries to show per category"
    )
    
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("""
    1. Select category below
    2. News is automatically:
       - 📡 Fetched from BBC
       - 🎯 Classified by AI
       - ✨ Summarized
       - 📊 Displayed
    """)
    
    st.markdown("---")
    
    # Check classifier status
    if st.session_state.classifier is None and NEURAL_AVAILABLE:
        st.session_state.classifier = load_classifier()
    
    if st.session_state.classifier is None:
        st.warning("⚠️ No trained model found. Train in 'Train Model' tab first!")
    else:
        st.success("✅ AI Classifier Ready")

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>📰 BBC News Summarizer & Classifier</h1>
    <p>
        Real-time BBC News → Structured summaries → AI Classification → Downloadable reports
    </p>
    <p>⚡ Fast • 🎯 Classify • 📄 Summarize • 📥 Download</p>
</div>
""", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📰 Live News", "🔍 Classify Article", "🎓 Train Model"])

# ---------------- TAB 1: LIVE NEWS ----------------
with tab1:
    st.markdown("### 📺 Select News Category")
    st.markdown("Click any category to automatically fetch, classify, and summarize the latest BBC news.")
    
    # 5 categories matching the trained model
    all_categories = {
        "business": {"icon": "💰", "desc": "Economy, markets, companies"},
        "entertainment": {"icon": "🎬", "desc": "Movies, TV, music, cinema"},
        "politics": {"icon": "🏛️", "desc": "Government, elections, policies"},
        "sport": {"icon": "⚽", "desc": "Football, cricket, Olympics"},
        "tech": {"icon": "💻", "desc": "Tech news, gadgets, AI"}
    }
    
    # Display categories as clickable cards
    cols = st.columns(5)
    for i, (cat_name, cat_info) in enumerate(all_categories.items()):
        with cols[i % 5]:
            if st.button(
                f"{cat_info['icon']} **{cat_name.upper()}**\n\n{cat_info['desc']}",
                key=f"cat_{cat_name}",
                use_container_width=True
            ):
                # Set selected category
                st.session_state.selected_category = cat_name
                # Clear previous data
                st.session_state.news_articles = []
                st.session_state.summaries = {}
                st.session_state.classifications = {}
                st.rerun()
    
    st.markdown("---")
    
    # Check if a category is selected
    if "selected_category" not in st.session_state:
        st.info("👆 **Select a category above** to get AI-summarized news automatically!")
        st.markdown("""
        ### ✨ What happens when you select a category:
        1. 📡 **Fetches** latest BBC news articles
        2. 🎯 **Classifies** each article using AI
        3. 📄 **Extracts** full article content
        4. ✨ **Generates** structured summaries
        5. 📊 **Displays** only articles matching your category
        
        **Note:** Make sure you've trained a model first (see 'Train Model' tab)
        """)
    else:
        selected_cat = st.session_state.selected_category
        cat_info = all_categories[selected_cat]
        
        st.markdown(f"## {cat_info['icon']} {selected_cat.upper()} News")
        st.markdown(f"*{cat_info['desc']}*")
        
        # Check if classifier is available
        if st.session_state.classifier is None:
            st.error("⚠️ No trained classifier found. Please train a model in the 'Train Model' tab first!")
            st.stop()
        
        # Auto-fetch and process if not already done
        if not st.session_state.news_articles:
            with st.spinner(f"🔄 Fetching and analyzing {selected_cat} news..."):
                # Step 1: Fetch ALL BBC articles (model will classify them)
                articles = fetch_all_bbc_news()
                
                # Step 2: Classify all articles
                progress_bar = st.progress(0)
                status = st.empty()
                
                classified_articles = []
                for i, article in enumerate(articles):
                    status.text(f"🎯 Classifying article {i+1}/{len(articles)}...")
                    progress_bar.progress((i + 1) / len(articles) * 0.5)  # First 50%
                    
                    text = f"{article['title']}. {article['description']}"
                    try:
                        pred_cat, confidence = st.session_state.classifier.predict_with_confidence(text)
                        
                        # Normalize category names (e.g. "sports" → "sport")
                        pred_cat = pred_cat.lower().rstrip("s") if pred_cat.lower() in ["sports"] else pred_cat.lower()
                        match_cat = selected_cat.lower()
                        
                        # Only keep articles matching selected category
                        if pred_cat == match_cat and confidence > 0.3:
                            classified_articles.append({
                                **article,
                                'category': pred_cat,
                                'confidence': confidence
                            })
                            
                            # Stop when we have enough
                            if len(classified_articles) >= num_articles:
                                break
                    except Exception:
                        pass
                
                # Step 3: Fetch full content and summarize
                for i, article in enumerate(classified_articles):
                    status.text(f"✨ Summarizing article {i+1}/{len(classified_articles)}...")
                    progress_bar.progress(0.5 + (i + 1) / len(classified_articles) * 0.5)  # Second 50%
                    
                    try:
                        full_text = fetch_full_article(article["url"])
                        if full_text:
                            raw_summary = summarize(full_text)
                            article['summary'] = format_summary(raw_summary)
                        else:
                            article['summary'] = "Summary unavailable - could not fetch article content."
                    except Exception:
                        article['summary'] = "Summary generation failed."
                
                st.session_state.news_articles = classified_articles
                progress_bar.empty()
                status.empty()
        
        # Display results
        if not st.session_state.news_articles:
            st.warning(f"😔 No {selected_cat} articles found in the latest BBC news. Try again later or select a different category.")
            if st.button("🔄 Try Again"):
                st.session_state.news_articles = []
                st.rerun()
        else:
            st.success(f"✅ Found **{len(st.session_state.news_articles)}** {selected_cat} articles with summaries!")
            
            # Display each article with summary
            for i, article in enumerate(st.session_state.news_articles):
                st.markdown('<div class="article-card">', unsafe_allow_html=True)
                
                st.markdown(f"### 📰 {article['title']}")
                st.markdown(f"🎯 **Category:** {article['category'].upper()} ({article['confidence']:.0%} confidence)")
                st.markdown(f"📅 **Published:** {article['publishedAt']}")
                st.markdown(f"🔗 [Read Full Article]({article['url']})")
                
                # Display summary
                st.markdown(f"<div class='summary-box'>{article['summary']}</div>", unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    "📥 Download Summary",
                    article['summary'],
                    file_name=f"{article['category']}_{i+1}_summary.txt",
                    mime="text/plain",
                    key=f"dl_art_{i}"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Refresh button
            if st.button("🔄 Refresh News", use_container_width=True):
                st.session_state.news_articles = []
                st.session_state.summaries = {}
                st.session_state.classifications = {}
                st.rerun()

# ---------------- TAB 2: CLASSIFY ARTICLE ----------------
with tab2:
    st.subheader("🔍 Classify Any Article")
    st.markdown("Enter article text below to classify it into a BBC News category.")
    
    # Load classifier
    if st.session_state.classifier is None and NEURAL_AVAILABLE:
        st.session_state.classifier = load_classifier()
    
    if not NEURAL_AVAILABLE:
        st.error("Classifier module not available. Make sure scikit-learn is installed.")
    elif st.session_state.classifier is None:
        st.warning("No trained model found. Please train a model first in the 'Train Model' tab.")
    else:
        input_text = st.text_area(
            "Article Text",
            height=200,
            placeholder="Paste your article text here..."
        )
        
        if st.button("🎯 Classify Text"):
            if input_text.strip():
                with st.spinner("Classifying..."):
                    pred_cat, confidence = st.session_state.classifier.predict_with_confidence(input_text)
                    
                    st.success(f"### Predicted Category: **{pred_cat.upper()}**")
                    st.info(f"Confidence: **{confidence:.1%}**")
                    
                    # Show all probabilities
                    st.markdown("#### All Category Probabilities:")
                    probs = st.session_state.classifier.predict_proba(input_text)
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    
                    for cat, prob in sorted_probs:
                        st.progress(prob, text=f"{cat}: {prob:.1%}")
            else:
                st.warning("Please enter some text to classify.")

# ---------------- TAB 3: TRAIN MODEL ----------------
with tab3:
    st.subheader("🎓 Train News Classifier")
    st.markdown("""
    Train a machine learning model to classify BBC News articles into categories.
    Choose between traditional ML (SVM, Naive Bayes) or Deep Learning (Neural Network).
    """)
    
    if not NEURAL_AVAILABLE:
        st.error("Neural Network module not available. Make sure PyTorch is installed.")
        st.code("pip install torch numpy scikit-learn", language="bash")
    else:
        # Data source selection
        st.markdown("### Step 1: Choose Data Source")
        data_source = st.radio(
            "Training Data Source",
            ["📁 Upload CSV Dataset (Recommended)", "📚 20 Newsgroups (Auto-download)"],
            help="CSV datasets give better results with more data"
        )
        
        if "📁" in data_source:
            st.info("""
            **Recommended Datasets:**
            - **BBC News Dataset**: [Download from Kaggle](https://www.kaggle.com/c/learn-ai-bbc/data) - 2,225 articles, 5 categories
            - **AG News Dataset**: [Download from Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) - 120K articles, 4 categories
            
            Place the CSV file in `sic/data/` folder.
            """)
            
            csv_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
            
            col1, col2 = st.columns(2)
            with col1:
                text_col = st.text_input("Text Column Name", value="text", help="Column containing article text")
            with col2:
                cat_col = st.text_input("Category Column Name", value="category", help="Column containing category labels")
        
        st.markdown("### Step 2: Configure Neural Network")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, help="Fraction of data for testing")
        with col2:
            epochs = st.slider("Epochs", min_value=5, max_value=30, value=10, help="Number of training iterations")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        with col2:
            learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
        
        st.info("""
        **Neural Network Architecture:**
        - Embedding Layer (128 dim) → Word representations
        - Bidirectional LSTM (256 hidden) → Sequence processing
        - FC Layer + ReLU → Feature extraction
        - FC Layer + ReLU → Feature extraction  
        - Softmax Output → Category probabilities
        """)
        
        st.markdown("### Step 3: Train")
        if st.button("🚀 Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize neural classifier
                classifier = NeuralNewsClassifier()
                
                # Load data based on source
                if "📁" in data_source:
                    if csv_file is None:
                        st.warning("Please upload a CSV file.")
                        st.stop()
                    
                    status_text.text("Loading CSV dataset...")
                    progress_bar.progress(10)
                    
                    # Save uploaded file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                        tmp.write(csv_file.getvalue())
                        tmp_path = tmp.name
                    
                    classifier.load_from_csv(tmp_path, text_column=text_col, category_column=cat_col, verbose=False)
                    os.unlink(tmp_path)
                    
                else:
                    status_text.text("Downloading 20 Newsgroups dataset...")
                    progress_bar.progress(10)
                    classifier.load_20newsgroups(verbose=False)
                
                progress_bar.progress(30)
                status_text.text(f"Loaded {len(classifier.training_data)} samples. Training Neural Network...")
                
                # Train neural network
                metrics = classifier.train(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    test_size=test_size,
                    verbose=False
                )
                
                progress_bar.progress(90)
                status_text.text("Saving model...")
                
                # Save model
                model_dir = os.path.join(script_dir, "models")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "neural_classifier.pt")
                classifier.save_model(model_path)
                
                # Update session state
                st.session_state.classifier = classifier
                
                progress_bar.progress(100)
                status_text.text("Training complete!")
                
                st.success(f"""
                ### ✅ Training Complete!
                - **Model Type:** Neural Network (LSTM)
                - **Training Samples:** {len(classifier.training_data)}
                - **Categories:** {', '.join(classifier.categories)}
                - **Accuracy:** {metrics['accuracy']:.1%}
                - **Model saved to:** `models/neural_classifier.pt`
                """)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        st.markdown("---")
        st.markdown("### Command Line Training")
        st.markdown("For more control, train from the command line:")
        st.code("""
# Train using 20 Newsgroups (auto-downloads)
python classifier/train_classifier.py --dataset 20news

# Train with more epochs
python classifier/train_classifier.py --epochs 15

# Train using BBC News CSV dataset
python classifier/train_classifier.py --dataset bbc

# Train using custom CSV
python classifier/train_classifier.py --csv data/mydata.csv --text-col content --cat-col label
        """, language="bash")
