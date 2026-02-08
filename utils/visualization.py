"""Visualization utilities for training analysis and model comparison."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation loss/accuracy curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        train_accs: List of training accuracies per epoch.
        val_accs: List of validation accuracies per epoch.
        save_path: Optional path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(preds, targets, class_names, save_path=None):
    """Plot a confusion matrix heatmap.

    Args:
        preds: Predicted class indices.
        targets: Ground truth class indices.
        class_names: List of class names.
        save_path: Optional path to save the figure.
    """
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_accuracy_vs_latency(model_names, accuracies, latencies, save_path=None):
    """Scatter plot of accuracy vs. inference latency for model comparison.

    Args:
        model_names: List of model names.
        accuracies: List of top-1 accuracies.
        latencies: List of inference latencies in ms.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(latencies, accuracies, s=100, zorder=5)

    for name, lat, acc in zip(model_names, latencies, accuracies):
        ax.annotate(name, (lat, acc), textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy vs. Latency Trade-off")
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
