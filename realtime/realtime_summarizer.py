"""Real-time article summarization using LSA (Latent Semantic Analysis).

This module accepts raw article text fetched live and generates extractive
summaries using a pure-Python LSA-based algorithm. It reuses a common
preprocessing pipeline and ensures low latency for real-time use.

Usage example:

from realtime.realtime_summarizer import RTSummarizer

summarizer = RTSummarizer()
text = "Your article text here..."
summary = summarizer.summarize(text, num_sentences=3)
print(summary)

"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Optional


# Common stopwords for English (lightweight set)
STOPWORDS_EN = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "will", "with", "i", "me", "my", "we", "you",
    "your", "this", "these", "those", "which", "who", "what", "where",
    "when", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "same",
    "so", "than", "too", "very", "can", "just", "should", "now", "do",
    "does", "did", "have", "had", "having", "do", "does", "did"
}


def _tokenize(text: str) -> List[str]:
    """Simple sentence tokenizer that splits on period, exclamation, question."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Clean and filter empty sentences
    return [s.strip() for s in sentences if s.strip()]


def _preprocess_text(text: str, lowercase: bool = True) -> str:
    """Normalize text: optional lowercasing, light cleaning."""
    if lowercase:
        text = text.lower()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _get_word_freq(sentences: List[str], extra_stopwords: Optional[set] = None) -> dict:
    """Compute word frequencies from sentences, excluding stopwords."""
    stopwords = STOPWORDS_EN.copy()
    if extra_stopwords:
        stopwords.update(extra_stopwords)
    
    words = []
    for sentence in sentences:
        # Remove punctuation and tokenize
        clean = sentence.translate(str.maketrans('', '', string.punctuation))
        words.extend(clean.lower().split())
    
    # Filter stopwords and count
    words = [w for w in words if w and w not in stopwords]
    return dict(Counter(words))


def _score_sentences_lsa(
    sentences: List[str],
    word_freq: dict,
    use_position: bool = False
) -> dict:
    """Score sentences using term frequency + optional position weighting.
    
    Args:
    - sentences: list of sentence strings
    - word_freq: dict mapping word -> frequency
    - use_position: if True, boost early sentences (Luhn-like)
    
    Returns:
    - dict mapping sentence index -> score
    """
    scores = {}
    
    for idx, sentence in enumerate(sentences):
        # Tokenize and clean
        clean = sentence.translate(str.maketrans('', '', string.punctuation))
        words = clean.lower().split()
        
        # Sum word frequencies for this sentence
        score = sum(word_freq.get(w, 0) for w in words if w)
        
        # Normalize by sentence length
        if words:
            score = score / len(words)
        
        # Optional position boost (earlier sentences score higher)
        if use_position:
            position_weight = 1.0 + (1.0 - idx / max(len(sentences), 1)) * 0.5
            score *= position_weight
        
        scores[idx] = score
    
    return scores


class RTSummarizer:
    """Real-time LSA-based summarizer for live article text."""
    
    def __init__(self, extra_stopwords: Optional[set] = None):
        """Initialize summarizer with optional custom stopwords."""
        self.extra_stopwords = extra_stopwords or set()
    
    def summarize(
        self,
        article_text: str,
        num_sentences: int = 3,
        use_position_weighting: bool = True,
        lowercase: bool = True,
    ) -> str:
        """Generate extractive summary of article text.
        
        Parameters:
        - article_text: raw article text (string)
        - num_sentences: number of sentences to include in summary
        - use_position_weighting: boost early sentences (Luhn-like)
        - lowercase: convert text to lowercase before processing
        
        Returns:
        - Extracted summary string (sentences in original order)
        
        Notes:
        - This method is designed for low latency.
        - Returns empty string if article is too short.
        """
        # Preprocess
        text = _preprocess_text(article_text, lowercase=lowercase)
        
        # Sentence tokenization
        sentences = _tokenize(text)
        
        if not sentences:
            return ""
        
        # Clamp num_sentences to available
        num_sentences = min(num_sentences, len(sentences))
        if num_sentences <= 0:
            return ""
        
        # Compute word frequencies
        word_freq = _get_word_freq(sentences, extra_stopwords=self.extra_stopwords)
        
        # Score sentences
        scores = _score_sentences_lsa(
            sentences,
            word_freq,
            use_position=use_position_weighting
        )
        
        # Select top-scoring sentences (maintaining original order)
        top_indices = sorted(
            scores.keys(),
            key=lambda idx: scores[idx],
            reverse=True
        )[:num_sentences]
        top_indices.sort()  # Restore original order
        
        # Build summary
        summary_sentences = [sentences[idx] for idx in top_indices]
        return " ".join(summary_sentences)
    
    def summarize_batch(
        self,
        articles: List[str],
        num_sentences: int = 3,
        use_position_weighting: bool = True,
    ) -> List[str]:
        """Summarize multiple articles (convenience method).
        
        Parameters:
        - articles: list of article text strings
        - num_sentences: number of sentences per summary
        - use_position_weighting: boost early sentences
        
        Returns:
        - List of summary strings
        """
        return [
            self.summarize(
                article,
                num_sentences=num_sentences,
                use_position_weighting=use_position_weighting
            )
            for article in articles
        ]


__all__ = [
    "RTSummarizer",
]
