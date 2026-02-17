# BBC News Summarizer & Classifier

A real-time news summarization and classification application powered by deep learning. Fetches live news from BBC RSS feeds, generates structured summaries, and classifies articles into categories using a neural network.

## Features

- **Live BBC News**: Fetches real-time news from BBC News RSS feeds
- **AI Summarization**: Generates structured summaries with key points, health risks, and actionable insights
- **Neural Network Classification**: LSTM-based deep learning classifier for news categorization
- **Interactive UI**: Built with Streamlit for easy use

## Supported Categories

- Politics, Technology, Sports, Health, Business
- Entertainment, Science, World, UK, Education

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sic
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Training the Neural Network

### Option 1: Via Web Interface
1. Run the app with `streamlit run app.py`
2. Go to the "Train Model" tab
3. Upload a CSV dataset or use 20 Newsgroups
4. Configure epochs, batch size, learning rate
5. Click "Start Training"

### Option 2: Via Command Line
```bash
# Train with 20 Newsgroups dataset (auto-downloads)
python classifier/train_classifier.py --dataset 20news

# Train with BBC News CSV
python classifier/train_classifier.py --dataset bbc

# Train with more epochs
python classifier/train_classifier.py --epochs 15

# Train with custom CSV
python classifier/train_classifier.py --csv data/mydata.csv --text-col content --cat-col label
```

### Available Options
```
--dataset, -d     Dataset: 20news, bbc, ag (default: 20news)
--csv             Path to custom CSV file
--epochs          Number of training epochs (default: 10)
--batch-size      Batch size (default: 32)
--learning-rate   Learning rate (default: 0.001)
--output, -o      Output path (default: models/neural_classifier.pt)
```

## Using the Classifier Programmatically

```python
from classifier.neural_classifier import NeuralNewsClassifier

# Load a pre-trained model
classifier = NeuralNewsClassifier()
classifier.load_model('models/neural_classifier.pt')

# Predict category
category = classifier.predict("Your article text here...")

# Get prediction with confidence
category, confidence = classifier.predict_with_confidence("Article text...")

# Get all category probabilities
probs = classifier.predict_proba("Article text...")
```

## Project Structure

```
sic/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── classifier/
│   ├── __init__.py
│   ├── neural_classifier.py    # Neural Network classifier (PyTorch LSTM)
│   └── train_classifier.py     # Command-line training script
├── data/                       # Training datasets (CSV files)
├── realtime/
│   └── realtime_summarizer.py  # Text summarization
└── models/                     # Saved trained models
```

## Technologies

- **Streamlit**: Web interface
- **PyTorch**: Deep Learning (LSTM Neural Network)
- **scikit-learn**: Data splitting and metrics
- **feedparser**: RSS feed parsing
- **BeautifulSoup**: Web scraping
- **requests**: HTTP requests

## License

MIT License
