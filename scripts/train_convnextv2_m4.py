#!/usr/bin/env python3
"""ConvNeXt-V2 Nano training on a small subset — optimized for MacBook Pro M4.

Expects data to already be prepared at data/subset/{train,val,test}/ by
running scripts/prepare_data.py first.

This script:
  1. Trains ConvNeXt-V2 Nano with MPS-optimized settings
  2. Validates per epoch with accuracy + macro F1
  3. Tests on held-out data with full classification report
  4. Saves learning curves, confusion matrix, and prediction grid

Usage:
    python3 scripts/prepare_data.py            # first: download & create subset
    python3 scripts/train_convnextv2_m4.py     # then: train
    python3 scripts/train_convnextv2_m4.py --epochs 20 --batch_size 32
"""

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 0. Ensure dependencies
# ---------------------------------------------------------------------------
def _ensure_package(package: str, pip_name: str = None):
    """Import a package, installing via pip if missing."""
    try:
        __import__(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"[setup] Installing {pip_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"]
        )


_ensure_package("timm")
_ensure_package("albumentations")
_ensure_package("sklearn", "scikit-learn")
_ensure_package("seaborn")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import get_train_transforms, get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from data.dataset import PlantDiseaseDataset
from models.convnextv2 import create_convnextv2
from utils.metrics import compute_accuracy, compute_macro_f1, compute_classification_report
from utils.visualization import plot_learning_curves, plot_confusion_matrix
from utils.benchmark import count_parameters, get_model_size_mb

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-V2 Nano (M4 optimized)")
    parser.add_argument("--subset_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "subset"),
                        help="Path to subset with train/val/test splits "
                             "(created by prepare_data.py).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train from scratch (no ImageNet weights).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Device detection
# ---------------------------------------------------------------------------
def get_device():
    """Select best available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        print("[device] Using Apple MPS (Metal Performance Shaders)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(f"[device] Using CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("[device] Using CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = running_loss / len(all_targets)
    acc = compute_accuracy(all_preds, all_targets)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model. Returns (avg_loss, accuracy, macro_f1, all_preds, all_targets)."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = running_loss / len(all_targets)
    acc = compute_accuracy(all_preds, all_targets)
    f1 = compute_macro_f1(all_preds, all_targets)
    return avg_loss, acc, f1, all_preds, all_targets


# ---------------------------------------------------------------------------
# 4. Visualization: prediction grid
# ---------------------------------------------------------------------------
def plot_prediction_grid(
    model, dataset, class_names, device, save_path, n_samples=16, seed=42
):
    """Plot a 4x4 grid of test images with true/predicted labels.

    Green title = correct, red title = incorrect.
    """
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for ax_idx, data_idx in enumerate(indices):
        image_tensor, label = dataset[data_idx]
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)
            confidence, pred = probs.max(dim=1)
            pred = pred.item()
            confidence = confidence.item()

        # De-normalize for display
        img_np = image_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        true_name = class_names[label]
        pred_name = class_names[pred]
        correct = pred == label

        axes[ax_idx].imshow(img_np)
        axes[ax_idx].set_title(
            f"True: {true_name}\nPred: {pred_name} ({confidence:.1%})",
            color="green" if correct else "red",
            fontsize=9,
        )
        axes[ax_idx].axis("off")

    for ax_idx in range(len(indices), len(axes)):
        axes[ax_idx].axis("off")

    plt.suptitle("ConvNeXt-V2 Nano — Sample Predictions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Prediction grid saved to {save_path}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()

    # Output directories
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Validate subset exists ----
    subset_dir = Path(args.subset_dir)
    for split in ["train", "val", "test"]:
        split_path = subset_dir / split
        if not split_path.exists():
            print(f"[error] Missing {split_path}")
            print("Run 'python3 scripts/prepare_data.py' first to download and create the subset.")
            sys.exit(1)

    # ---- Build datasets and loaders ----
    train_transform = get_train_transforms(args.image_size)
    val_transform = get_val_transforms(args.image_size)

    train_dataset = PlantDiseaseDataset(
        os.path.join(args.subset_dir, "train"), transform=train_transform
    )
    val_dataset = PlantDiseaseDataset(
        os.path.join(args.subset_dir, "val"), transform=val_transform
    )
    test_dataset = PlantDiseaseDataset(
        os.path.join(args.subset_dir, "test"), transform=val_transform
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"\n[data] Classes ({num_classes}): {class_names}")
    print(f"[data] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # MPS-optimized DataLoader settings
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=0,       # MPS + multiprocess has macOS issues
        pin_memory=False,    # Not supported on MPS
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # ---- Model ----
    pretrained = not args.no_pretrained
    print(f"\n[model] Creating ConvNeXt-V2 Nano (pretrained={pretrained})")
    model = create_convnextv2(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    n_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print(f"[model] Parameters: {n_params:,}")
    print(f"[model] Size: {size_mb:.2f} MB")

    # ---- Training setup ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3), T_mult=2, eta_min=1e-6
    )

    # ---- Training loop ----
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_f1": [],
    }
    best_f1 = 0.0
    best_ckpt = ckpt_dir / "convnextv2_nano_best.pth"

    print(f"\n{'='*60}")
    print(f" Training ConvNeXt-V2 Nano — {args.epochs} epochs, bs={args.batch_size}")
    print(f" Device: {device} | LR: {args.lr} | Weight Decay: {args.weight_decay}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f} | "
            f"LR: {lr_now:.2e} | {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
            }, best_ckpt)
            print(f"  -> Saved best model (F1={val_f1:.4f})")

    # ---- Load best model for testing ----
    print(f"\n{'='*60}")
    print(" Testing with best model")
    print(f"{'='*60}\n")

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[test] Loaded best checkpoint from epoch {ckpt['epoch']} (F1={ckpt['val_f1']:.4f})")

    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\n[test] Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  Macro-F1: {test_f1:.4f}\n")

    report = compute_classification_report(test_preds, test_targets, class_names)
    print(report)

    # ---- Visualizations ----
    print(f"\n{'='*60}")
    print(" Generating visualizations")
    print(f"{'='*60}\n")

    lc_path = str(fig_dir / "convnextv2_learning_curves.png")
    plot_learning_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"], history["val_acc"],
        save_path=lc_path,
    )
    print(f"[viz] Learning curves saved to {lc_path}")

    cm_path = str(fig_dir / "convnextv2_confusion_matrix.png")
    plot_confusion_matrix(test_preds, test_targets, class_names, save_path=cm_path)
    print(f"[viz] Confusion matrix saved to {cm_path}")

    pred_path = str(fig_dir / "convnextv2_predictions.png")
    plot_prediction_grid(
        model, test_dataset, class_names, device, save_path=pred_path,
    )

    print(f"\n{'='*60}")
    print(" Done!")
    print(f"  Best checkpoint: {best_ckpt}")
    print(f"  Figures: {fig_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
