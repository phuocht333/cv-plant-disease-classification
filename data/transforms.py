"""Image preprocessing and augmentation pipelines.

Uses Albumentations for augmentations with CLAHE (Contrast Limited Adaptive
Histogram Equalization) to handle greenhouse fog/LED lighting conditions.

CLAHE is applied deterministically (p=1.0) in both train and val as a
preprocessing step for greenhouse image correction.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default augmentation config (used when no config dict is provided)
DEFAULT_AUG_CFG = {
    "clahe": {
        "clip_limit": 2.0,
        "tile_grid_size": [8, 8],
    },
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.5,
    "rotate90_p": 0.5,
    "shift_scale_rotate": {
        "shift_limit": 0.1,
        "scale_limit": 0.15,
        "rotate_limit": 15,
        "p": 0.5,
    },
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.5,
    },
    "gauss_noise_p": 0.3,
    "gaussian_blur_p": 0.2,
}


def _get_clahe(aug_cfg: dict) -> A.CLAHE:
    """Build CLAHE transform from config (always p=1.0)."""
    clahe_cfg = aug_cfg.get("clahe", DEFAULT_AUG_CFG["clahe"])
    return A.CLAHE(
        clip_limit=clahe_cfg.get("clip_limit", 2.0),
        tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
        p=1.0,
    )


def get_train_transforms(image_size: int = 224, aug_cfg: dict = None):
    """Build training augmentation pipeline.

    Includes CLAHE (p=1.0) for greenhouse image enhancement, random
    augmentations, and normalization with ImageNet statistics.

    Args:
        image_size: Target image size (default 224).
        aug_cfg: Augmentation config dict (from default.yaml 'augmentation' section).
                 Falls back to DEFAULT_AUG_CFG if None.

    Returns:
        Albumentations Compose transform.
    """
    if aug_cfg is None:
        aug_cfg = DEFAULT_AUG_CFG

    ssr = aug_cfg.get("shift_scale_rotate", DEFAULT_AUG_CFG["shift_scale_rotate"])
    cj = aug_cfg.get("color_jitter", DEFAULT_AUG_CFG["color_jitter"])

    return A.Compose([
        A.Resize(image_size, image_size),
        # Greenhouse preprocessing — always applied
        _get_clahe(aug_cfg),
        # Random augmentations (training only)
        A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
        A.VerticalFlip(p=aug_cfg.get("vertical_flip_p", 0.5)),
        A.RandomRotate90(p=aug_cfg.get("rotate90_p", 0.5)),
        A.ShiftScaleRotate(
            shift_limit=ssr.get("shift_limit", 0.1),
            scale_limit=ssr.get("scale_limit", 0.15),
            rotate_limit=ssr.get("rotate_limit", 15),
            p=ssr.get("p", 0.5),
        ),
        A.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
            saturation=cj.get("saturation", 0.2),
            hue=cj.get("hue", 0.1),
            p=cj.get("p", 0.5),
        ),
        A.GaussNoise(p=aug_cfg.get("gauss_noise_p", 0.3)),
        A.GaussianBlur(blur_limit=(3, 7), p=aug_cfg.get("gaussian_blur_p", 0.2)),
        # Normalization + tensor conversion
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224, aug_cfg: dict = None):
    """Build validation transform pipeline.

    Applies CLAHE (p=1.0) and normalization only (no random augmentations).

    Args:
        image_size: Target image size (default 224).
        aug_cfg: Augmentation config dict for CLAHE parameters.
                 Falls back to DEFAULT_AUG_CFG if None.

    Returns:
        Albumentations Compose transform.
    """
    if aug_cfg is None:
        aug_cfg = DEFAULT_AUG_CFG

    return A.Compose([
        A.Resize(image_size, image_size),
        _get_clahe(aug_cfg),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# Alias — test transforms are identical to validation
get_test_transforms = get_val_transforms
