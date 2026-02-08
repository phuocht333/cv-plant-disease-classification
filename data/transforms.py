"""Image preprocessing and augmentation pipelines.

Uses Albumentations for augmentations with CLAHE (Contrast Limited Adaptive
Histogram Equalization) to handle greenhouse fog/LED lighting conditions.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224):
    """Build training augmentation pipeline.

    Includes CLAHE for greenhouse image enhancement, standard augmentations,
    and normalization with ImageNet statistics.

    Args:
        image_size: Target image size (default 224).

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    """Build validation/test transform pipeline.

    Applies CLAHE and normalization only (no random augmentations).

    Args:
        image_size: Target image size (default 224).

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
