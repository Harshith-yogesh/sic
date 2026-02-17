"""Neural Network News Classifier using PyTorch.

This module provides a deep learning-based text classification model with:
- Embedding layer for word representations
- LSTM/GRU layers for sequence processing
- Fully connected layers with ReLU activation
- Softmax output for category probabilities

Architecture:
    Input Text → Tokenizer → Embedding → LSTM → FC(ReLU) → FC(ReLU) → Softmax → Category

Usage:
    from classifier.neural_classifier import NeuralNewsClassifier
    
    classifier = NeuralNewsClassifier()
    classifier.load_data_from_csv('data/bbc-text.csv')
    classifier.train(epochs=10)
    classifier.save_model('models/neural_classifier.pt')
    
    category = classifier.predict("Article text here...")
"""
from __future__ import annotations

import os
import pickle
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Sklearn for data splitting and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class TextClassifierNN(nn.Module):
    """Neural Network for Text Classification.
    
    Architecture:
        Embedding → LSTM → Dropout → FC(ReLU) → Dropout → Output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 5,
        n_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.3)  # Dropout on embeddings
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # FC layer with ReLU activation + heavy dropout
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(64, output_dim)
        
    def forward(self, text, text_lengths):
        # Embedding + dropout
        embedded = self.embedding(text)
        embedded = self.embed_dropout(embedded)
        
        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Final hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # FC + ReLU + Dropout
        x = self.fc1(hidden)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        output = self.fc_out(x)
        return output


# ============================================================================
# DATASET CLASS
# ============================================================================

class NewsDataset(Dataset):
    """PyTorch Dataset for news articles."""
    
    def __init__(self, texts: List[List[int]], labels: List[int]):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def collate_batch(batch):
    """Custom collate function for variable length sequences."""
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return texts_padded, labels, lengths


# ============================================================================
# MAIN CLASSIFIER CLASS
# ============================================================================

class NeuralNewsClassifier:
    """Neural Network-based News Classifier.
    
    This classifier uses:
    - Word tokenization and vocabulary building
    - Word embeddings (learned during training)
    - LSTM for sequence processing
    - ReLU activation functions
    - Softmax output layer
    - Cross-entropy loss with Adam optimizer
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.5,
        max_vocab_size: int = 15000,
        max_seq_length: int = 300
    ):
        """Initialize the neural classifier.
        
        Args:
            embed_dim: Word embedding dimension
            hidden_dim: LSTM hidden dimension
            n_layers: Number of LSTM layers
            dropout: Dropout probability
            max_vocab_size: Maximum vocabulary size
            max_seq_length: Maximum sequence length
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        
        self.model: Optional[TextClassifierNN] = None
        self.vocab: Dict[str, int] = {}
        self.label_encoder: Dict[str, int] = {}
        self.label_decoder: Dict[int, str] = {}
        self.training_data: List[Dict] = []
        self.is_trained = False
        self.categories: List[str] = []
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        return tokens[:self.max_seq_length]
    
    def _build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
        
        # Most common words
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # Reserve 0 for padding, 1 for unknown
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(most_common, start=2):
            self.vocab[word] = idx
            
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def _encode_text(self, text: str) -> List[int]:
        """Convert text to list of token indices."""
        tokens = self._tokenize(text)
        return [self.vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
    
    def _build_label_encoder(self, labels: List[str]):
        """Build label encoder/decoder."""
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        self.categories = unique_labels
        print(f"Categories: {self.categories}")
    
    def add_training_data(self, text: str, category: str):
        """Add a training sample."""
        self.training_data.append({
            "text": text.strip(),
            "category": category.lower()
        })
    
    def load_from_csv(
        self,
        filepath: str,
        text_column: str = "text",
        category_column: str = "category",
        verbose: bool = True
    ) -> List[Dict]:
        """Load training data from CSV."""
        import csv
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        if verbose:
            print(f"Loading dataset from {filepath}...")
        
        self.training_data = []
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(text_column, "").strip()
                category = row.get(category_column, "").strip().lower()
                
                if text and category and len(text) > 50:
                    self.training_data.append({
                        "text": text,
                        "category": category
                    })
        
        if verbose:
            print(f"Loaded {len(self.training_data)} samples")
        
        return self.training_data
    
    def load_20newsgroups(self, verbose: bool = True) -> List[Dict]:
        """Load 20 Newsgroups dataset."""
        from sklearn.datasets import fetch_20newsgroups
        
        categories = [
            'talk.politics.misc',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.med',
            'comp.sys.mac.hardware',
            'sci.space',
            'rec.autos'
        ]
        
        category_map = {
            'talk.politics.misc': 'politics',
            'rec.sport.baseball': 'sports',
            'rec.sport.hockey': 'sports',
            'sci.med': 'health',
            'comp.sys.mac.hardware': 'technology',
            'sci.space': 'science',
            'rec.autos': 'entertainment',
        }
        
        if verbose:
            print("Downloading 20 Newsgroups dataset...")
        
        newsgroups = fetch_20newsgroups(
            subset='train',
            categories=categories,
            remove=('headers', 'footers', 'quotes')
        )
        
        self.training_data = []
        
        for text, target in zip(newsgroups.data, newsgroups.target):
            original_cat = newsgroups.target_names[target]
            category = category_map.get(original_cat, 'other')
            
            if text.strip() and len(text) > 50:
                self.training_data.append({
                    "text": text.strip(),
                    "category": category
                })
        
        if verbose:
            print(f"Loaded {len(self.training_data)} samples")
        
        return self.training_data
    
    def train(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """Train the neural network.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            test_size: Fraction of data for testing
            verbose: Print training progress
            
        Returns:
            Dictionary with training metrics
        """
        if not self.training_data:
            raise ValueError("No training data. Load data first.")
        
        # Prepare data
        texts = [d["text"] for d in self.training_data]
        labels = [d["category"] for d in self.training_data]
        
        # Build vocabulary and label encoder
        self._build_vocab(texts)
        self._build_label_encoder(labels)
        
        # Encode texts and labels
        encoded_texts = [self._encode_text(t) for t in texts]
        encoded_labels = [self.label_encoder[l] for l in labels]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_texts, encoded_labels, 
            test_size=test_size, 
            random_state=42,
            stratify=encoded_labels
        )
        
        # Create datasets and dataloaders
        train_dataset = NewsDataset(X_train, y_train)
        test_dataset = NewsDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_batch
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            collate_fn=collate_batch
        )
        
        # Initialize model
        self.model = TextClassifierNN(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=len(self.categories),
            n_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=False
        ).to(self.device)
        
        if verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {total_params:,}", flush=True)
        
        # Loss function and optimizer with weight decay (L2 regularization)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler - reduce LR when accuracy plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        if verbose:
            print(f"\nTraining on {self.device}", flush=True)
            print(f"Model architecture:", flush=True)
            print(self.model, flush=True)
            print(f"\nTraining samples: {len(X_train)}", flush=True)
            print(f"Test samples: {len(X_test)}", flush=True)
            print("-" * 50, flush=True)
        
        # Training loop with early stopping
        best_accuracy = 0
        best_model_state = None
        patience = 5
        patience_counter = 0
        history = {'train_loss': [], 'test_accuracy': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = len(train_loader)
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs} training...", end="", flush=True)
            
            for batch_idx, (texts_batch, labels_batch, lengths) in enumerate(train_loader):
                texts_batch = texts_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                lengths = lengths.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(texts_batch, lengths)
                loss = criterion(predictions, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                
                if verbose and (batch_idx + 1) % 10 == 0:
                    print(".", end="", flush=True)
            
            avg_loss = total_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            # Evaluate
            test_accuracy = self._evaluate(test_loader)
            history['test_accuracy'].append(test_accuracy)
            
            # Step the scheduler
            scheduler.step(test_accuracy)
            
            # Early stopping - save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {test_accuracy:.2%} | Best: {best_accuracy:.2%} | LR: {current_lr:.6f}", flush=True)
            
            # Early stop if no improvement for 'patience' epochs
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping! No improvement for {patience} epochs.", flush=True)
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"\nRestored best model with {best_accuracy:.2%} accuracy.", flush=True)
        
        self.is_trained = True
        
        # Final evaluation
        if verbose:
            print("\n" + "=" * 50, flush=True)
            print("Final Evaluation:", flush=True)
            print("=" * 50, flush=True)
            self._detailed_evaluation(test_loader)
        
        return {
            "accuracy": best_accuracy,
            "final_accuracy": history['test_accuracy'][-1],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "epochs": epochs,
            "model_type": "neural_network"
        }
    
    def _evaluate(self, data_loader) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts_batch, labels_batch, lengths in data_loader:
                texts_batch = texts_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                lengths = lengths.to(self.device)
                
                predictions = self.model(texts_batch, lengths)
                _, predicted = torch.max(predictions, 1)
                
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        return correct / total
    
    def _detailed_evaluation(self, data_loader):
        """Print detailed classification report."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts_batch, labels_batch, lengths in data_loader:
                texts_batch = texts_batch.to(self.device)
                lengths = lengths.to(self.device)
                
                predictions = self.model(texts_batch, lengths)
                _, predicted = torch.max(predictions, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.numpy())
        
        # Convert to category names
        pred_names = [self.label_decoder[p] for p in all_preds]
        label_names = [self.label_decoder[l] for l in all_labels]
        
        print(classification_report(label_names, pred_names))
    
    def predict(self, text: str) -> str:
        """Predict category for a text.
        
        Args:
            text: Article text to classify
            
        Returns:
            Predicted category string
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        self.model.eval()
        
        # Encode text
        encoded = self._encode_text(text)
        if not encoded:
            return self.categories[0]  # Default category
        
        # Create tensor
        text_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
        length_tensor = torch.tensor([len(encoded)]).to(self.device)
        
        with torch.no_grad():
            output = self.model(text_tensor, length_tensor)
            _, predicted = torch.max(output, 1)
        
        return self.label_decoder[predicted.item()]
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """Predict category probabilities.
        
        Args:
            text: Article text to classify
            
        Returns:
            Dictionary mapping categories to probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained.")
        
        self.model.eval()
        
        encoded = self._encode_text(text)
        if not encoded:
            return {cat: 1.0/len(self.categories) for cat in self.categories}
        
        text_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
        length_tensor = torch.tensor([len(encoded)]).to(self.device)
        
        with torch.no_grad():
            output = self.model(text_tensor, length_tensor)
            # Apply softmax to get probabilities
            probs = torch.softmax(output, dim=1)[0]
        
        return {self.label_decoder[i]: probs[i].item() for i in range(len(self.categories))}
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Predict category with confidence score.
        
        Args:
            text: Article text to classify
            
        Returns:
            Tuple of (predicted_category, confidence)
        """
        probs = self.predict_proba(text)
        best_category = max(probs, key=probs.get)
        confidence = probs[best_category]
        return best_category, confidence
    
    def save_model(self, filepath: str):
        """Save model and associated data."""
        if not self.is_trained or self.model is None:
            raise ValueError("No trained model to save.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Save model state and metadata
        data = {
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'categories': self.categories,
            'config': {
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
                'max_vocab_size': self.max_vocab_size,
                'max_seq_length': self.max_seq_length,
                'vocab_size': len(self.vocab),
                'output_dim': len(self.categories)
            }
        }
        
        torch.save(data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        data = torch.load(filepath, map_location=self.device)
        
        # Restore metadata
        self.vocab = data['vocab']
        self.label_encoder = data['label_encoder']
        self.label_decoder = data['label_decoder']
        self.categories = data['categories']
        
        config = data['config']
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        
        # Rebuild model
        self.model = TextClassifierNN(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval()
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        print(f"Categories: {self.categories}")
        print(f"Device: {self.device}")


__all__ = ["NeuralNewsClassifier", "TextClassifierNN"]
