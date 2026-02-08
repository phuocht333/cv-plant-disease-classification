"""Main training script for plant disease classification.

Usage:
    python scripts/train.py --model mobilenetv4_conv_small --data_dir /path/to/data
    python scripts/train.py --model convnextv2_nano --config configs/default.yaml
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import PlantDiseaseDataset
from data.transforms import get_train_transforms, get_val_transforms
from models import get_model
from utils.metrics import compute_accuracy, compute_macro_f1


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch.

    Args:
        model: The model to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for mixed precision.
        device: Device to train on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = compute_accuracy(all_preds, all_targets)
    return avg_loss, acc


def validate(model, loader, criterion, device):
    """Validate the model.

    Args:
        model: The model to evaluate.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        Tuple of (average_loss, accuracy, macro_f1).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", leave=False):
            images, targets = images.to(device), targets.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = compute_accuracy(all_preds, all_targets)
    f1 = compute_macro_f1(all_preds, all_targets)
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (mobilenetv4_conv_small, convnextv2_nano, ghostnetv3)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root (with train/ and val/ subdirectories)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_transform = get_train_transforms(cfg["data"]["image_size"])
    val_transform = get_val_transforms(cfg["data"]["image_size"])

    train_dataset = PlantDiseaseDataset(
        os.path.join(args.data_dir, "train"), transform=train_transform
    )
    val_dataset = PlantDiseaseDataset(
        os.path.join(args.data_dir, "val"), transform=val_transform
    )

    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # Model
    model = get_model(args.model, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["loss"]["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["scheduler"]["T_0"],
        T_mult=cfg["scheduler"]["T_mult"],
        eta_min=cfg["scheduler"]["eta_min"],
    )
    scaler = GradScaler()

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    # Training loop
    best_f1 = 0.0
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(
                cfg["output"]["checkpoint_dir"], f"{args.model}_best.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  Saved best model (F1={val_f1:.4f}) to {ckpt_path}")

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
