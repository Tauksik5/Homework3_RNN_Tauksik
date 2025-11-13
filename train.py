import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.models import get_model
from src.preprocess import load_and_preprocess, get_loaders

# fixing the random seeds
import random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    preds, labels_list = [], []
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds.extend((outputs > 0.5).int().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_list, preds)
    f1 = f1_score(labels_list, preds)
    avg_loss = total_loss / len(dataloader)
    return acc, f1, avg_loss

def train_model(
    model_name="rnn",
    seq_length=50,
    optimizer_name="adam",
    lr=0.001,
    epochs=3,
    batch_size=32,
    grad_clip=False,
):
    # loading and preparing the data
    x_train, y_train, x_test, y_test = load_and_preprocess(seq_length=seq_length)
    train_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test, batch_size)

    vocab_size = 10000
    model = get_model(model_name, vocab_size).to(device)
    criterion = nn.BCELoss()

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")

    print(f"Training {model_name.upper()} model for {epochs} epochs...")
    start_time = time.time()

    train_losses = []
    val_losses = []


    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer)

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        acc, f1, val_loss = evaluate(model, test_loader, criterion)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
        )

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")

    # Save loss curves for analysis
    os.makedirs("results/plots", exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, marker='o', label='Validation Loss')
    plt.title(f"{model_name.upper()} Loss vs Epochs (Seq={seq_length}, Opt={optimizer_name.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_{optimizer_name}_seq{seq_length}_loss_curve.png")
    plt.close()
    return model

import csv
import os

def save_results(model_name, optimizer, seq_length, grad_clip, acc, f1, epoch_time):
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "metrics.csv")

    header = ["Model", "Optimizer", "Seq_Length", "Grad_Clipping", "Accuracy", "F1", "Epoch_Time"]
    data = [model_name, optimizer, seq_length, grad_clip, round(acc, 4), round(f1, 4), round(epoch_time, 2)]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)

if __name__ == "__main__":
    import time

    configs = []
    models = ["rnn", "lstm", "bilstm"]
    seq_lengths = [25, 50, 100]
    optimizers = ["adam", "sgd", "rmsprop"]

    # building all the combinations automatically
    for m in models:
        for s in seq_lengths:
            for opt in optimizers:
                configs.append((m, opt, s, True))

    for name, opt, seq_len, clip in configs:
        print(f"\n----- Training {name.upper()} | Seq={seq_len} | Opt={opt.upper()} -----")
        start = time.time()
        model = train_model(
            model_name=name,
            seq_length=seq_len,
            optimizer_name=opt,
            epochs=5,          # running 5 epochs for each model
            grad_clip=clip
        )
        duration = time.time() - start

        # evaluating after the training
        x_train, y_train, x_test, y_test = load_and_preprocess(seq_length=seq_len)
        _, test_loader = get_loaders(x_train, y_train, x_test, y_test)
        criterion = nn.BCELoss()
        acc, f1, _ = evaluate(model, test_loader, criterion)

        save_results(name.upper(), opt.upper(), seq_len, clip, acc, f1, duration)
        print(f"{name.upper()} | Seq={seq_len} | Opt={opt.upper()} done. "
              f"Acc={acc:.4f}, F1={f1:.4f}, Time={duration:.2f}s")