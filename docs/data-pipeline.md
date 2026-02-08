# Data Pipeline Implementation — CLAHE Preprocessing & Albumentations Integration

## Problem

The skeleton data pipeline had a critical bug and several missing features:

1. **Albumentations compatibility bug**: `PlantDiseaseDataset.__getitem__` passed a PIL Image directly to Albumentations transforms. Albumentations expects a numpy array via the `transform(image=np_array)` dict API — this would crash at runtime with a `TypeError`.
2. CLAHE was set to `p=0.5` in training, meaning only half the training images received greenhouse lighting correction — inconsistent preprocessing.
3. All augmentation parameters were hardcoded magic numbers, not configurable from the YAML config.
4. No file extension filtering — non-image files (`.txt`, `.DS_Store`, etc.) in class folders would cause `Image.open()` failures.
5. No class distribution analysis for diagnosing class imbalance.
6. No dataloader factory — every script had to manually wire up transforms → datasets → dataloaders (~20 lines of boilerplate).

## Changes

### 1. `data/dataset.py` — Fix + harden the Dataset class

**Albumentations fix (critical):**

```python
# BEFORE (broken) — passes PIL Image, wrong API
image = Image.open(img_path).convert("RGB")
if self.transform:
    image = self.transform(image)  # TypeError at runtime

# AFTER (correct) — converts to numpy, uses dict API
image = Image.open(img_path).convert("RGB")
image = np.array(image)  # HWC uint8 numpy array
if self.transform:
    image = self.transform(image=image)["image"]  # Albumentations dict pattern
```

**Why PIL → numpy?** Albumentations operates on numpy arrays (OpenCV convention: HWC, uint8). The `transform(image=...)` dict call returns a dict where `["image"]` contains the result. This is Albumentations' core API contract — every transform in the Compose pipeline receives and returns numpy arrays, and `ToTensorV2()` at the end converts to a PyTorch tensor.

**Image extension filtering:**

```python
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
```

Added in `_load_samples()` to skip non-image files. Without this, files like `.DS_Store` (macOS), `thumbs.db` (Windows), or stray `.txt` files in the dataset directory would cause PIL to fail with `UnidentifiedImageError`. Using a set for O(1) lookup.

**`get_class_distribution()` method:**

Returns `{class_name: count}` using `collections.Counter`. This is needed for:
- Diagnosing class imbalance (common in plant disease datasets)
- Deciding whether to apply class-weighted sampling or oversampling
- Reporting dataset statistics in experiment logs

### 2. `data/transforms.py` — Config-driven augmentation pipeline

**CLAHE always applied (p=1.0):**

```python
A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)  # both train AND val
```

**Reasoning:** CLAHE is not a random augmentation — it's a deterministic preprocessing step to correct greenhouse-specific imaging conditions (fog, uneven LED lighting, low contrast). It must be applied consistently to every image in both training and validation to ensure the model sees the same preprocessed input distribution. Setting `p=0.5` in training (as the original code did) would mean the model trains on a mix of corrected and uncorrected images, creating a train/val distribution mismatch.

**Config-driven parameters:**

Every augmentation probability and parameter is now read from a config dict:

```python
def get_train_transforms(image_size=224, aug_cfg=None):
    aug_cfg = aug_cfg or DEFAULT_AUG_CFG
    # All params from aug_cfg with sensible defaults
    A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5))
    ...
```

**Why?** Hardcoded augmentation values make hyperparameter tuning impossible without code changes. With config-driven params, you can run ablation studies (e.g., CLAHE clip_limit sweep, augmentation probability tuning) by just changing `configs/default.yaml` or passing experiment-specific YAML overrides — no code edits needed.

**Added GaussNoise and GaussianBlur:**

```python
A.GaussNoise(p=0.3)
A.GaussianBlur(blur_limit=(3, 7), p=0.2)
```

**Why?** Greenhouse cameras often produce noisy images (low-cost sensors, variable lighting). Adding synthetic noise and blur during training improves robustness to these real-world conditions. The probabilities are conservative (0.3 and 0.2) to avoid degrading image quality too aggressively.

**`get_test_transforms` alias:**

```python
get_test_transforms = get_val_transforms
```

Test-time preprocessing should be identical to validation — no random augmentations, just resize + CLAHE + normalize. The alias makes this explicit in calling code without duplicating the function.

### 3. `data/__init__.py` — Dataloader factory

```python
def create_dataloaders(data_dir, cfg):
    """Returns dict with train_loader, val_loader, num_classes, classes, class_to_idx"""
```

**Why a factory?** The manual setup pattern (create transforms → create datasets → create dataloaders → extract metadata) was ~20 lines of boilerplate repeated wherever data loading was needed. The factory:

- Reads augmentation config from the YAML automatically
- Instantiates both datasets with correct transforms
- Creates DataLoaders with config-driven batch_size, num_workers, pin_memory
- Prints dataset statistics (class count, sample counts, distribution) for experiment logging
- Returns all metadata (num_classes, class names) needed by downstream code

This follows the principle of a single source of truth — data loading logic lives in one place, not scattered across scripts.

### 4. `configs/default.yaml` — Augmentation config section

```yaml
augmentation:
  clahe:
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  horizontal_flip_p: 0.5
  vertical_flip_p: 0.5
  rotate90_p: 0.5
  shift_scale_rotate:
    shift_limit: 0.1
    scale_limit: 0.15
    rotate_limit: 15
    p: 0.5
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p: 0.5
  gauss_noise_p: 0.3
  gaussian_blur_p: 0.2
```

Grouped logically: CLAHE preprocessing params, geometric augmentations (flips, rotation, affine), photometric augmentations (color jitter), and noise/blur augmentations. Each augmentation's probability is independently tunable for ablation studies.

### 5. `scripts/train.py` — Simplified

```python
# BEFORE: ~20 lines of manual setup
train_transform = get_train_transforms(cfg["data"]["image_size"])
val_transform = get_val_transforms(cfg["data"]["image_size"])
train_dataset = PlantDiseaseDataset(os.path.join(args.data_dir, "train"), ...)
val_dataset = PlantDiseaseDataset(os.path.join(args.data_dir, "val"), ...)
num_classes = len(train_dataset.classes)
train_loader = DataLoader(train_dataset, batch_size=..., shuffle=True, ...)
val_loader = DataLoader(val_dataset, batch_size=..., shuffle=False, ...)

# AFTER: 4 lines
data = create_dataloaders(args.data_dir, cfg)
train_loader = data["train_loader"]
val_loader = data["val_loader"]
num_classes = data["num_classes"]
```

## Augmentation Pipeline Summary

| Stage | Train | Val/Test | Rationale |
|-------|-------|----------|-----------|
| Resize 224x224 | Always | Always | Model input requirement |
| CLAHE (p=1.0) | Always | Always | Greenhouse lighting correction |
| HorizontalFlip | p=0.5 | No | Leaves are symmetric |
| VerticalFlip | p=0.5 | No | Camera angle variation |
| RandomRotate90 | p=0.5 | No | Orientation invariance |
| ShiftScaleRotate | p=0.5 | No | Geometric robustness |
| ColorJitter | p=0.5 | No | Lighting variation |
| GaussNoise | p=0.3 | No | Sensor noise robustness |
| GaussianBlur | p=0.2 | No | Focus variation robustness |
| Normalize (ImageNet) | Always | Always | Pretrained model compatibility |
| ToTensorV2 | Always | Always | PyTorch format (CHW float32) |

## Verification

All checks pass with a synthetic 3-class dataset:
- `__getitem__` returns `(tensor[3, 224, 224], int)` — correct shape and types
- Non-image files (`.txt`) are properly filtered out
- `get_class_distribution()` returns accurate counts
- `create_dataloaders()` returns working DataLoaders that yield correct batch dimensions
