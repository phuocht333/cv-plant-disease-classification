# Notebook Pipeline Summary

Two self-contained Jupyter notebooks that take a raw Kaggle dataset (YOLO detection format) through to a trained ConvNeXt-V2 Nano classifier with full evaluation and visualizations.

---

## Notebook 01 — Data Preparation & Transforms

**File:** `01_prepare_data.ipynb`

**Goal:** Turn a raw Kaggle dataset (YOLO detection format) into a clean, class-per-folder image subset ready for classification training.

### Steps

| Step | What | Rationale |
|------|------|-----------|
| 1. Setup | `pip install kaggle albumentations pyyaml` | Ensures dependencies exist on Colab (fresh environment each session) |
| 2. Project root | Auto-detect project root by searching for `data/` or `configs/` | Works whether running from `notebooks/` subfolder or Colab `/content/` |
| 3. Config | `DataConfig` dataclass — paths, subset sizes (50/15/10 per class), CLAHE params, seed | Single source of truth; easy to adjust without touching logic |
| 4. Download | Kaggle API downloads & unzips `ironwolf437/plant-disease-detection-dataset` into `data/raw/` | Idempotent — skips if data already exists. Uses Python API (not CLI) because `python3 -m kaggle` is broken on some runtimes |
| 5. Organise | Parse `data.yaml` for 7 class names, read each YOLO `.txt` label to get dominant class ID, copy images into `data/organised/{train,val,test}/{class_name}/` | The raw dataset is in YOLO detection format (images/ + labels/ per split) — classification training needs class-per-folder layout. Uses `Counter.most_common(1)` when an image has multiple bounding boxes |
| 6. Subset | Random sample N images per class into `data/subset/{train,val,test}/` (350 train / 105 val / 70 test) | Full dataset is ~40K images — too large for quick experiments. Fixed seed ensures reproducibility |
| 7. Transforms | `TransformFactory` builds two Albumentations pipelines: **Train** (CLAHE + augmentations + Normalize) and **Val/Test** (CLAHE + Normalize only) | CLAHE at p=1.0 corrects greenhouse fog/LED lighting artifacts. Augmentations prevent overfitting on the small subset. ImageNet normalization matches pretrained backbone expectations |
| 8. Dataset & Loaders | `PlantDiseaseDataset` scans class folders, loads images as numpy arrays, applies Albumentations | Albumentations expects numpy input (not PIL), so conversion happens inside `__getitem__` |
| 9. Verify | 2x4 grid of de-normalized training images + class distribution bar chart | Visual sanity check that transforms look reasonable and classes are balanced (50 per class) |

### Transform Pipelines

| Pipeline | Transforms |
|----------|------------|
| **Train** | Resize(224) → CLAHE(p=1.0) → HFlip → VFlip → Rotate90 → ShiftScaleRotate → ColorJitter → GaussNoise → GaussianBlur → Normalize → ToTensor |
| **Val/Test** | Resize(224) → CLAHE(p=1.0) → Normalize → ToTensor |

### Outputs

```
data/organised/{train,val,test}/{class_name}/   # Full dataset, class-per-folder
data/subset/{train,val,test}/{class_name}/       # Small subset for quick training
```

### Dataset Stats

- **7 classes:** downy mildew, early blight, gray mold, green mold, late blight, powdery mildew, tomato mosaic virus
- **Full organised:** 39,097 train / 1,633 val / 1,625 test
- **Subset:** 350 train / 105 val / 70 test (50/15/10 per class)

---

## Notebook 02 — Train ConvNeXt-V2 Nano

**File:** `02_train_convnextv2.ipynb`

**Goal:** Fine-tune a pretrained ConvNeXt-V2 Nano on the subset, evaluate on held-out test data, and produce all required visualizations.

**Prerequisite:** Run notebook 01 first to create `data/subset/`.

### Runtime Selector

Set `RUNTIME = "mac"` or `RUNTIME = "colab"` at the top of the config cell (default: `"mac"`).

| Setting | Mac (MPS) | Colab (CUDA) |
|---------|-----------|--------------|
| Device | `mps` | `cuda` |
| Batch size | 16 | 32 |
| FP16 mixed precision | Off | On |
| num_workers | 0 | 2 |
| pin_memory | False | True |

### Steps

| Step | What | Rationale |
|------|------|-----------|
| 1. Setup | `pip install timm albumentations` | `timm` provides pretrained ConvNeXt-V2 via `timm.create_model()` |
| 2. Project root | Same root-finder as notebook 01 + asserts `data/subset/` exists | Fails early with a helpful message if user forgot to run notebook 01 |
| 3. Runtime selector | `RUNTIME = "mac"` or `"colab"` with presets dict | One variable to flip — all device-specific settings follow automatically |
| 4. Config | `TrainConfig` dataclass — reads defaults from runtime preset, plus LR=1e-4, AdamW weight_decay=0.05, label_smoothing=0.1, cosine annealing schedule | Centralised hyperparameters. LR=1e-4 (not 1e-3) avoids destroying pretrained features on a small dataset |
| 5. Device | Selects `cuda`/`mps`/`cpu` based on `RUNTIME`, falls back to CPU with warning | Explicit choice prevents confusion vs silent auto-detection |
| 6. Dataset & Loaders | Re-defines `PlantDiseaseDataset` + `TransformFactory`, builds train/val/test loaders | Self-contained — notebook 02 runs independently without importing from notebook 01 |
| 7. Model | `timm.create_model("convnextv2_nano", pretrained=True, num_classes=7)` | Pretrained ImageNet backbone + fresh classifier head. ~15M params, 57MB |
| 8. Trainer | `Trainer` class with AdamW, CosineAnnealingWarmRestarts, LabelSmoothing CE, optional FP16 | Per-epoch checkpointing for Colab disconnect resilience (see below) |
| 9. Train | `trainer.train()` — loops epochs, logs metrics, saves checkpoints | Prints `*BEST*` marker when val F1 improves |
| 10. Evaluate | Loads best checkpoint, runs inference on test set, prints full `classification_report` | Best-by-val-F1 model (not latest) — standard practice to avoid reporting overfit results |
| 11. Visualize | Learning curves, confusion matrix heatmap, 4x4 prediction grid (green=correct, red=wrong) | Required outputs per project specification |
| 12. Summary | Prints model stats: param count, size, epochs, best val F1, test accuracy/F1 | Quick reference for comparing against other architectures |

### Checkpoint & Resume Strategy

| File | Content | Purpose |
|------|---------|---------|
| `latest_checkpoint.pth` | Model + optimizer + scheduler + scaler + history + epoch | Resume training after disconnect — re-run the training cell |
| `convnextv2_nano_best.pth` | Same state dict, saved when val F1 improves | Used by Evaluator for final test metrics |

### Training Recipe

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW (lr=1e-4, weight_decay=0.05) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-6) |
| Epochs | 10 (subset training) |
| Mixed precision | FP16 on CUDA only, disabled on MPS |

### Outputs

```
outputs/checkpoints/convnextv2_nano_best.pth        # Best model weights
outputs/checkpoints/latest_checkpoint.pth            # Resumable training state
outputs/figures/convnextv2_learning_curves.png       # Train/val loss & accuracy
outputs/figures/convnextv2_confusion_matrix.png      # True vs predicted heatmap
outputs/figures/convnextv2_predictions.png           # 4x4 sample prediction grid
```

---

## OOP Class Overview

### Notebook 01

| Class | Methods | Responsibility |
|-------|---------|----------------|
| `DataConfig` | (dataclass) | All data-related settings |
| `DataPreparer` | `download()`, `organise()`, `create_subset()`, `run()` | Full data pipeline (download → YOLO conversion → subset) |
| `PlantDiseaseDataset` | `_scan()`, `__getitem__()` | PyTorch Dataset for class-per-folder images |
| `TransformFactory` | `train()`, `val()` | Albumentations pipeline builder |

### Notebook 02

| Class | Methods | Responsibility |
|-------|---------|----------------|
| `TrainConfig` | (dataclass) | All training settings (reads from runtime preset) |
| `PlantDiseaseDataset` | `_scan()`, `__getitem__()` | Same as notebook 01 (self-contained) |
| `TransformFactory` | `train()`, `val()` | Same as notebook 01 (self-contained) |
| `ModelFactory` | `create()`, `count_parameters()`, `get_size_mb()` | Model creation via `timm` + inspection |
| `Trainer` | `_save_checkpoint()`, `_try_resume()`, `_train_one_epoch()`, `_evaluate()`, `train()` | Training loop with checkpoint save/resume |
| `Evaluator` | `load_best_checkpoint()`, `evaluate()` | Test-set evaluation with classification report |
| `Visualizer` | `plot_learning_curves()`, `plot_confusion_matrix()`, `plot_prediction_grid()` | All figure generation |
