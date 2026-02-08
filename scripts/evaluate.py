"""Evaluate a trained model on the test set.

Usage:
    python scripts/evaluate.py --model mobilenetv4_conv_small \
        --checkpoint outputs/checkpoints/mobilenetv4_conv_small_best.pth \
        --data_dir /path/to/data/test
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import PlantDiseaseDataset
from data.transforms import get_val_transforms
from models import get_model
from utils.metrics import compute_accuracy, compute_macro_f1, compute_classification_report
from utils.visualization import plot_confusion_matrix
from utils.benchmark import count_parameters, get_model_size_mb, measure_latency, measure_throughput


def main():
    parser = argparse.ArgumentParser(description="Evaluate plant disease classifier")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to test dataset directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    transform = get_val_transforms(cfg["data"]["image_size"])
    dataset = PlantDiseaseDataset(args.data_dir, transform=transform)
    num_classes = len(dataset.classes)

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    # Model
    model = get_model(args.model, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Inference
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            with autocast():
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics
    acc = compute_accuracy(all_preds, all_targets)
    f1 = compute_macro_f1(all_preds, all_targets)
    report = compute_classification_report(all_preds, all_targets, dataset.classes)

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model Size: {get_model_size_mb(model):.2f} MB")
    print(f"\nClassification Report:\n{report}")

    # Benchmark
    if device.type == "cuda":
        latency = measure_latency(model, device=str(device))
        throughput = measure_throughput(model, device=str(device))
        print(f"Inference Latency: {latency:.2f} ms")
        print(f"Throughput: {throughput:.1f} img/s")

    # Confusion matrix
    save_path = os.path.join(cfg["output"]["figure_dir"], f"{args.model}_confusion_matrix.png")
    plot_confusion_matrix(all_preds, all_targets, dataset.classes, save_path=save_path)
    print(f"\nConfusion matrix saved to {save_path}")


if __name__ == "__main__":
    main()
