import pandas as pd
import matplotlib.pyplot as plt
import os

# reading the metrics csv file
df = pd.read_csv("results/metrics.csv")

# make sure plots directory exists
os.makedirs("results/plots", exist_ok=True)

# accuracy and f1 score by sequence length for each model
def plot_accuracy_f1(df):
    models = df["Model"].unique()
    for model in models:
        subset = df[(df["Model"] == model) & (df["Optimizer"] == "ADAM")]
        plt.figure(figsize=(6, 4))
        plt.plot(subset["Seq_Length"], subset["Accuracy"], marker="o", label="Accuracy")
        plt.plot(subset["Seq_Length"], subset["F1"], marker="o", label="F1-score")
        plt.title(f"{model} - Accuracy and F1 vs Sequence Length (Adam)")
        plt.xlabel("Sequence Length")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/plots/{model}_accuracy_f1_vs_seq.png")
        plt.close()

# runtime comparison for each model and optimizer
def plot_training_time(df):
    plt.figure(figsize=(8, 5))
    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        plt.plot(subset["Seq_Length"], subset["Epoch_Time"], marker="o", label=model)
    plt.title("Training Time vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Epoch Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/training_time_vs_seq.png")
    plt.close()

def plot_accuracy_by_optimizer(df):
    models = df["Model"].unique()
    for model in models:
        subset = df[df["Model"] == model]
        pivot = subset.groupby("Optimizer")["Accuracy"].mean().reset_index()
        plt.figure(figsize=(5, 4))
        plt.bar(pivot["Optimizer"], pivot["Accuracy"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.title(f"{model} - Accuracy by Optimizer")
        plt.xlabel("Optimizer")
        plt.ylabel("Average Accuracy")
        plt.ylim(0, 1)
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"results/plots/{model}_accuracy_by_optimizer.png")
        plt.close()

def plot_f1_by_optimizer(df):
    models = df["Model"].unique()
    for model in models:
        subset = df[df["Model"] == model]
        pivot = subset.groupby("Optimizer")["F1"].mean().reset_index()
        plt.figure(figsize=(5, 4))
        plt.bar(pivot["Optimizer"], pivot["F1"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.title(f"{model} - F1-score by Optimizer")
        plt.xlabel("Optimizer")
        plt.ylabel("Average F1-score")
        plt.ylim(0, 1)
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"results/plots/{model}_f1_by_optimizer.png")
        plt.close()

if __name__ == "__main__":
    plot_accuracy_f1(df)
    plot_training_time(df)
    plot_accuracy_by_optimizer(df)
    plot_f1_by_optimizer(df)
    print("Plots saved in results/plots/")
