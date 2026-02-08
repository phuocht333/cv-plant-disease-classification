#!/usr/bin/env python3
"""Download the Kaggle dataset and create a small train/val/test subset.

The Kaggle dataset (ironwolf437/plant-disease-detection-dataset) is in YOLO
detection format (images/ + labels/ per split). This script:
  1. Downloads & unzips from Kaggle (if not already present)
  2. Parses data.yaml for class names
  3. Reads YOLO labels to determine each image's class
  4. Organises images into class-per-folder layout
  5. Samples a small subset into data/subset/{train,val,test}/

Requires ~/.kaggle/kaggle.json to be configured.

Usage:
    python3 scripts/prepare_data.py
    python3 scripts/prepare_data.py --train_per_class 100 --val_per_class 30 --test_per_class 20
    python3 scripts/prepare_data.py --force
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Ensure kaggle is installed
# ---------------------------------------------------------------------------
def _ensure_package(package: str, pip_name: str = None):
    try:
        __import__(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"[setup] Installing {pip_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"]
        )


_ensure_package("kaggle")
_ensure_package("yaml", "pyyaml")

import yaml


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Kaggle dataset and create a small subset"
    )
    parser.add_argument(
        "--raw_dir", type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Where to download/extract the raw YOLO-format dataset.",
    )
    parser.add_argument(
        "--organised_dir", type=str,
        default=str(PROJECT_ROOT / "data" / "organised"),
        help="Where to place images organised by class folders.",
    )
    parser.add_argument(
        "--subset_dir", type=str,
        default=str(PROJECT_ROOT / "data" / "subset"),
        help="Output directory for the small subset.",
    )
    parser.add_argument("--train_per_class", type=int, default=50)
    parser.add_argument("--val_per_class", type=int, default=15)
    parser.add_argument("--test_per_class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-create organised dir and subset even if they already exist.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "ironwolf437/plant-disease-detection-dataset"


def download_dataset(raw_dir: str):
    """Download and unzip the Kaggle dataset into raw_dir."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "plant-disease-detection-dataset.zip"

    if not zip_path.exists() and not any(raw_dir.iterdir()):
        print(f"[download] Downloading from Kaggle: {KAGGLE_DATASET}")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=False)
    else:
        print(f"[download] Dataset already present in {raw_dir}")

    if zip_path.exists():
        print(f"[download] Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()
        print("[download] Unzip complete, removed zip file.")


# ---------------------------------------------------------------------------
# Parse YOLO format → class-per-folder layout
# ---------------------------------------------------------------------------
def _parse_class_names(raw_dir: Path):
    """Read class names from data.yaml."""
    yaml_path = raw_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {raw_dir}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    print(f"[yolo] {len(names)} classes: {names}")
    return names


def _get_image_class(label_path: Path):
    """Read a YOLO label file and return the dominant class id.

    Each line: class_id x_center y_center width height
    Returns the most frequent class_id, or None if file is empty/missing.
    """
    if not label_path.exists():
        return None
    class_ids = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids.append(int(parts[0]))
    if not class_ids:
        return None
    return Counter(class_ids).most_common(1)[0][0]


def organise_yolo_to_class_folders(raw_dir: str, organised_dir: str, force: bool = False):
    """Convert YOLO format (images/ + labels/) to class-per-folder layout.

    Handles the dataset's split dirs: train/, valid/, test/.
    Maps them to: organised_dir/train/, organised_dir/val/, organised_dir/test/.
    """
    raw_dir = Path(raw_dir)
    organised_dir = Path(organised_dir)

    if not force and organised_dir.exists() and any(organised_dir.iterdir()):
        print(f"[organise] Already organised at {organised_dir}, skipping. Use --force to redo.")
        return

    if force and organised_dir.exists():
        print(f"[organise] Removing existing {organised_dir}")
        shutil.rmtree(organised_dir)

    class_names = _parse_class_names(raw_dir)

    # Map raw split names to output split names
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for raw_split, out_split in split_map.items():
        images_dir = raw_dir / raw_split / "images"
        labels_dir = raw_dir / raw_split / "labels"

        if not images_dir.exists():
            print(f"[organise] Skipping {raw_split}/ — no images/ dir")
            continue

        counts = Counter()
        skipped = 0

        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue

            label_file = labels_dir / (img_file.stem + ".txt")
            class_id = _get_image_class(label_file)

            if class_id is None or class_id >= len(class_names):
                skipped += 1
                continue

            class_name = class_names[class_id]
            dst = organised_dir / out_split / class_name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, dst / img_file.name)
            counts[class_name] += 1

        total = sum(counts.values())
        print(f"[organise] {out_split}: {total} images ({skipped} skipped, no label)")
        for cls in sorted(counts):
            print(f"    {cls}: {counts[cls]}")


# ---------------------------------------------------------------------------
# Subset creation (from class-per-folder layout)
# ---------------------------------------------------------------------------
def create_subset(
    organised_dir: str,
    subset_dir: str,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int = 42,
    force: bool = False,
):
    """Sample a small subset from the organised class-folder dataset."""
    subset_dir = Path(subset_dir)
    organised_dir = Path(organised_dir)

    if not force and (subset_dir / "train").exists() and (subset_dir / "val").exists():
        existing = list((subset_dir / "train").iterdir())
        if existing:
            print(f"[subset] Already exists at {subset_dir}, skipping. Use --force to redo.")
            _print_summary(subset_dir)
            return

    if force and subset_dir.exists():
        print(f"[subset] Removing existing {subset_dir}")
        shutil.rmtree(subset_dir)

    rng = random.Random(seed)
    limits = {"train": train_n, "val": val_n, "test": test_n}

    for split, n_per_class in limits.items():
        split_src = organised_dir / split
        if not split_src.exists():
            print(f"[subset] Skipping {split} — not found in organised dir")
            continue

        classes = sorted(
            d.name for d in split_src.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        for cls in classes:
            cls_dir = split_src / cls
            images = sorted(
                f.name for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )

            if len(images) <= n_per_class:
                sampled = images
            else:
                sampled = rng.sample(images, n_per_class)

            dst = subset_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for img_name in sampled:
                shutil.copy2(cls_dir / img_name, dst / img_name)

    _print_summary(subset_dir)


def _print_summary(subset_dir: Path):
    """Print image count per split."""
    print(f"\n[subset] Summary ({subset_dir}):")
    for split in ["train", "val", "test"]:
        split_path = subset_dir / split
        if not split_path.exists():
            print(f"  {split}: (not found)")
            continue
        n_images = sum(
            len(list((split_path / c).iterdir()))
            for c in os.listdir(split_path) if (split_path / c).is_dir()
        )
        n_classes = len([
            c for c in os.listdir(split_path)
            if (split_path / c).is_dir()
        ])
        print(f"  {split}: {n_images} images across {n_classes} classes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Step 1: Download if needed
    download_dataset(args.raw_dir)

    # Step 2: Convert YOLO format → class-per-folder
    organise_yolo_to_class_folders(args.raw_dir, args.organised_dir, force=args.force)

    # Step 3: Sample subset
    create_subset(
        organised_dir=args.organised_dir,
        subset_dir=args.subset_dir,
        train_n=args.train_per_class,
        val_n=args.val_per_class,
        test_n=args.test_per_class,
        seed=args.seed,
        force=args.force,
    )

    print("\n[done] Data preparation complete.")
    print(f"  Organised: {args.organised_dir}")
    print(f"  Subset:    {args.subset_dir}")
    print(f"  Train with: python3 scripts/train_convnextv2_m4.py")


if __name__ == "__main__":
    main()
