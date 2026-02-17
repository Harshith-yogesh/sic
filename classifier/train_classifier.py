"""Training script for the Neural Network News Classifier.

This script provides a command-line interface to:
1. Load training data from CSV datasets
2. Train the neural network classification model
3. Save the trained model for later use

Usage:
    python train_classifier.py --dataset 20news  # Train using 20 Newsgroups (auto-download)
    python train_classifier.py --dataset bbc     # Train using BBC News CSV dataset
    python train_classifier.py --csv path/to/file.csv  # Train using custom CSV
"""
import argparse
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from classifier.neural_classifier import NeuralNewsClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Train a Neural Network News Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from pre-built datasets:
  python train_classifier.py --dataset 20news        # Use 20 Newsgroups (auto-downloads)
  python train_classifier.py --dataset bbc           # Use BBC News CSV (place in data/BBC News Train.csv)
  python train_classifier.py --dataset ag            # Use AG News CSV (place in data/train.csv)
  
  # Train from custom CSV:
  python train_classifier.py --csv data/my_data.csv --text-col content --cat-col label
  
  # Training options:
  python train_classifier.py --epochs 15             # More training epochs
  python train_classifier.py --batch-size 64         # Larger batch size
  python train_classifier.py --learning-rate 0.0005  # Different learning rate
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["bbc", "ag", "20news"],
        default="20news",
        help="Dataset to use: bbc (BBC News CSV), ag (AG News), 20news (20 Newsgroups - default)"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a custom CSV file for training"
    )
    
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Column name for text in custom CSV (default: text)"
    )
    
    parser.add_argument(
        "--cat-col",
        type=str,
        default="category",
        help="Column name for category in custom CSV (default: category)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/neural_classifier.pt",
        help="Output path for the trained model (default: models/neural_classifier.pt)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs (default: 8)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    args = parser.parse_args()
    
    # Determine data source
    data_source = args.dataset if args.dataset else ("csv" if args.csv else "20news")
    
    print("=" * 60)
    print("Neural Network News Classifier Training")
    print("=" * 60)
    print(f"Data source: {data_source}")
    print(f"Output model: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Initialize neural classifier
    classifier = NeuralNewsClassifier()
    
    # Load data based on source
    if data_source == "bbc":
        csv_path = args.csv or os.path.join(parent_dir, "data", "BBC News Train.csv")
        if not os.path.exists(csv_path):
            print(f"\nERROR: BBC dataset not found at {csv_path}")
            print("\nTo download:")
            print("1. Go to: https://www.kaggle.com/c/learn-ai-bbc/data")
            print("2. Download BBC News Train.csv")
            print(f"3. Place it at: {csv_path}")
            return
        print(f"\nLoading BBC News dataset from {csv_path}...")
        classifier.load_from_csv(csv_path, text_column="Text", category_column="Category", verbose=True)
    
    elif data_source == "ag":
        csv_path = args.csv or os.path.join(parent_dir, "data", "train.csv")
        if not os.path.exists(csv_path):
            print(f"\nERROR: AG News dataset not found at {csv_path}")
            print("\nTo download:")
            print("1. Go to: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset")
            print("2. Download train.csv")
            print(f"3. Place it at: {csv_path}")
            return
        print(f"\nLoading AG News dataset from {csv_path}...")
        classifier.load_from_csv(csv_path, verbose=True)
    
    elif data_source == "20news":
        print("\nLoading 20 Newsgroups dataset (auto-download)...")
        classifier.load_20newsgroups(verbose=True)
    
    elif args.csv:
        print(f"\nLoading custom dataset from {args.csv}...")
        classifier.load_from_csv(
            args.csv,
            text_column=args.text_col,
            category_column=args.cat_col,
            verbose=True
        )
    
    else:
        print("\nERROR: No data source specified.")
        print("Use --dataset or --csv to specify training data.")
        return
    
    # Train model
    print("\n" + "=" * 60)
    print("Training Neural Network...")
    print("=" * 60 + "\n")
    
    metrics = classifier.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        verbose=True
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    classifier.save_model(args.output)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print("\nTo use the model:")
    print("  from classifier.neural_classifier import NeuralNewsClassifier")
    print("  classifier = NeuralNewsClassifier()")
    print(f"  classifier.load_model('{args.output}')")
    print("  category = classifier.predict('Your article text here...')")


if __name__ == "__main__":
    main()
