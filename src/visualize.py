"""Visualization tools for architecture comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_comparison_bars(df, metric, title, save_dir="output"):
    """Plot bar chart comparing architectures on a metric."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["name"], df[metric], color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric}_comparison.png"), dpi=150)
    plt.close()


def plot_accuracy_vs_params(results, save_dir="output"):
    """Scatter plot of accuracy vs parameter count."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in results.items():
        ax.scatter(data.get("params", 0) / 1e6, data.get("accuracy", 0),
                  s=100, label=name, zorder=5)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_params.png"), dpi=150)
    plt.close()
