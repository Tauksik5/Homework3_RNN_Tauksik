# Homework 3 — RNN, LSTM and BiLSTM for Sentiment Classification

## Overview
This project compares three recurrent neural network architectures — RNN, LSTM and Bidirectional LSTM for binary sentiment classification on the IMDb movie review dataset.  
The goal is to evaluate how architecture, optimizer and sequence length affect model performance and training efficiency.

## Setup and Execution

### 1. Installation
```bash
pip install -r requirements.txt
```
### 2. Run Training
Use this command to train all models and generate the results:

``` bash
python3 -m src.train
```
This script will:
- Preprocess the IMDb dataset  
- Train all the models automatically  
- Log the results to `results/metrics.csv`  
- Save all the generated plots inside this folder `results/plots/`

### Dataset and Preprocessing
- Dataset: IMDb Movie Review Dataset (50,000 samples)
- Split: 25,000 for training and 25,000 for testing
- Preprocessing steps:
  - Lowercase all text
  - Remove punctuation and special characters
  - Tokenize using NLTK `word_tokenize`
  - Keep top 10,000 most frequent words
  - Convert text to token IDs
  - Pad or truncate to fixed sequence lengths of 25, 50 and 100 words

### Model Architecture
- Embedding layer (size = 100)
- Two hidden layers (hidden size = 64)
- Dropout (0.3–0.5) for regularization
- Batch size = 32
- Fully connected output with sigmoid activation
- Binary cross-entropy loss
- Optimizers tested: Adam, SGD, RMSProp
- Gradient clipping for stability
- Trained for 5 epochs
- ReLU is used in the hidden layers (in model definitions)
- Sigmoid function is used in the final output layer for binary classification.
- Tanh may be used internally in the LSTM/RNN cells but PyTorch does this automatically inside the LSTM/RNN

### Results Summary

| Model | Activation | Optimizer | Seq Length | Grad Clipping | Accuracy | F1 Score | Epoch Time (s) |
|--------|-------------|------------|-------------|----------------|-----------|-----------|----------------|
| RNN | ReLU | Adam | 25 | Yes | 0.6846 | 0.6989 | 48.9 |
| RNN | ReLU | RMSProp | 25 | Yes | 0.6793 | 0.6731 | 45.5 |
| RNN | ReLU | SGD | 25 | Yes | 0.5051 | 0.4890 | 44.7 |
| RNN | ReLU | Adam | 50 | Yes | 0.6175 | 0.6426 | 57.8 |
| RNN | ReLU | RMSProp | 50 | Yes | 0.5818 | 0.6195 | 58.2 |
| RNN | ReLU | SGD | 50 | Yes | 0.5019 | 0.5075 | 53.4 |
| RNN | ReLU | Adam | 100 | Yes | 0.6275 | 0.6530 | 79.4 |
| RNN | ReLU | RMSProp | 100 | Yes | 0.6571 | 0.6466 | 75.2 |
| RNN | ReLU | SGD | 100 | Yes | 0.5053 | 0.5380 | 75.1 |
| LSTM | ReLU | Adam | 25 | Yes | 0.7229 | 0.7117 | 93.8 |
| LSTM | ReLU | RMSProp | 25 | Yes | 0.7124 | 0.6821 | 89.5 |
| LSTM | ReLU | SGD | 25 | Yes | 0.4981 | 0.6585 | 85.7 |
| LSTM | ReLU | Adam | 50 | Yes | 0.7676 | 0.7611 | 141.6 |
| LSTM | ReLU | RMSProp | 50 | Yes | 0.7732 | 0.7809 | 139.4 |
| LSTM | ReLU | SGD | 50 | Yes | 0.5051 | 0.4419 | 131.4 |
| LSTM | ReLU | Adam | 100 | Yes | 0.8139 | 0.8165 | 230.5 |
| LSTM | ReLU | RMSProp | 100 | Yes | 0.8098 | 0.8265 | 218.5 |
| LSTM | ReLU | SGD | 100 | Yes | 0.5016 | 0.6336 | 223.3 |
| BiLSTM | ReLU | Adam | 25 | Yes | 0.7262 | 0.7394 | 142.0 |
| BiLSTM | ReLU | RMSProp | 25 | Yes | 0.7193 | 0.7073 | 143.4 |
| BiLSTM | ReLU | SGD | 25 | Yes | 0.4954 | 0.4840 | 132.8 |
| BiLSTM | ReLU | Adam | 50 | Yes | 0.7761 | 0.7785 | 230.5 |
| BiLSTM | ReLU | RMSProp | 50 | Yes | 0.7345 | 0.7776 | 238.1 |
| BiLSTM | ReLU | SGD | 50 | Yes | 0.5016 | 0.5008 | 224.3 |
| BiLSTM | ReLU | Adam | 100 | Yes | 0.8156 | 0.8299 | 419.8 |
| BiLSTM | ReLU | RMSProp | 100 | Yes | 0.8165 | 0.8059 | 414.9 |
| BiLSTM | ReLU | SGD | 100 | Yes | 0.5048 | 0.4879 | 402.1 |

**Best Model:** BiLSTM with RMSProp (Seq = 100) — Accuracy = 0.8165, F1 = 0.8059  
**Worst Model:** RNN with SGD (Seq = 25) — Accuracy = 0.5051, F1 = 0.489

### Observations
- LSTM and BiLSTM outperform the RNNs due to better long-term dependency modeling.  
- RMSProp gives the best F1 scores while Adam optimiser converges faster.  
- Increasing the sequence length improves accuracy but also increases the training time.  
- Gradient clipping helps to prevent exploding gradients and stabilizes training.  
- LSTM and BiLSTM with longer sequences consistently yield the best results.

### Visualizations
All plots are saved under:
```bash
results/plots/
```

The generated plots include:
- Accuracy and F1 vs Sequence Length  
- Training Time vs Sequence Length  
- Training Loss vs Epochs (best and worst models)  
- Accuracy/F1 by Optimizer  

### Reproducibility

```python
import torch, random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```
Hardware used: MacBook Pro (M3 Pro, CPU mode only)

### Folder Structure

```css
Homework3_RNN_Tauksik/
│
├── data/
│   └── IMDB Dataset.csv
│
├── results/
│   ├── plots/
│   └── metrics.csv
│
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── utils.py
│
├── venv/
│   ├── bin/
│   ├── include/
│   ├── lib/
│   └── ...
│
├── requirements.txt
└── README.md
```

## Author
Name: Tauksik Anil Kumar

UID: 121331298

Course: DATA641 — Natural Language Processing  
