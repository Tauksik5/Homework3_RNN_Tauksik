import os
import ssl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Forcing NLTK to use my local punkt folder
nltk.data.path.append(os.path.expanduser("~/nltk_data"))

# Fixing the SSL error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Trying to load the punkt locally or download if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# fixing random seeds for reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def load_and_preprocess(path='data/IMDB Dataset.csv', seq_length=50, top_k=10000):
    print("Loading dataset...")
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Cleaning text...")
    df['review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    print("Tokenizing...")
    all_words = [w for r in df['review'] for w in word_tokenize(r)]
    word_freq = Counter(all_words)
    vocab = {w: i + 1 for i, (w, _) in enumerate(word_freq.most_common(top_k))}

    def encode(review):
        return [vocab[w] for w in word_tokenize(review) if w in vocab]

    encoded_reviews = [encode(r) for r in df['review']]

    print("Padding and truncating sequences...")
    padded = np.zeros((len(encoded_reviews), seq_length), dtype=int)
    for i, seq in enumerate(encoded_reviews):
        padded[i, :min(len(seq), seq_length)] = seq[:seq_length]

    split_idx = int(len(padded) * 0.5)
    x_train, y_train = padded[:split_idx], df['sentiment'][:split_idx].values
    x_test, y_test = padded[split_idx:], df['sentiment'][split_idx:].values

    print("Preprocessing completed.")
    return (
        torch.tensor(x_train), torch.tensor(y_train),
        torch.tensor(x_test), torch.tensor(y_test)
    )

def get_loaders(x_train, y_train, x_test, y_test, batch_size=32):
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
    return train_loader, test_loader

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_and_preprocess()
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")