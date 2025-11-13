import torch
import torch.nn as nn

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for reproducibility
torch.manual_seed(42)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=64, num_layers=2, dropout=0.4):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=64, num_layers=2, dropout=0.4):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        out, (h, c) = self.lstm(embedded)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=64, num_layers=2, dropout=0.4):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        out, (h, c) = self.lstm(embedded)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def get_model(name, vocab_size, embed_size=100, hidden_size=64, num_layers=2, dropout=0.4):
    name = name.lower()
    if name == "rnn":
        return RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout)
    elif name == "lstm":
        return LSTMModel(vocab_size, embed_size, hidden_size, num_layers, dropout)
    elif name == "bilstm":
        return BiLSTMModel(vocab_size, embed_size, hidden_size, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model name: {name}")
