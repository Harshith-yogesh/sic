"""Neural Network News Classifier Module.

This module provides a deep learning text classification model for news categories
using LSTM neural network with PyTorch.

Usage:
    from classifier import NeuralNewsClassifier
    
    classifier = NeuralNewsClassifier()
    classifier.load_model('models/neural_classifier.pt')
    category, confidence = classifier.predict_with_confidence("Article text...")
"""
from classifier.neural_classifier import NeuralNewsClassifier

__all__ = [
    "NeuralNewsClassifier"
]
