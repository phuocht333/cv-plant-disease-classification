"""Data pipeline for plant disease classification.

Provides a single-call factory ``create_dataloaders`` that handles transforms,
datasets, and DataLoader construction from a config dict.
"""

from torch.utils.data import DataLoader

from .dataset import PlantDiseaseDataset
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms

__all__ = [
    "PlantDiseaseDataset",
    "get_train_transforms",
    "get_val_transforms",
    "get_test_transforms",
    "create_dataloaders",
]


def create_dataloaders(data_dir: str, cfg: dict):
    """Create train and validation DataLoaders from a dataset directory.

    Expects ``data_dir`` to contain ``train/`` and ``val/`` subdirectories,
    each following the class-per-folder layout.

    Args:
        data_dir: Root dataset path containing ``train/`` and ``val/`` folders.
        cfg: Full config dict (as loaded from ``configs/default.yaml``).

    Returns:
        dict with keys:
            - ``train_loader``: Training DataLoader
            - ``val_loader``: Validation DataLoader
            - ``num_classes``: int
            - ``classes``: list of class name strings
            - ``class_to_idx``: dict mapping class name → index
    """
    import os

    image_size = cfg["data"]["image_size"]
    aug_cfg = cfg.get("augmentation", None)

    train_transform = get_train_transforms(image_size, aug_cfg=aug_cfg)
    val_transform = get_val_transforms(image_size, aug_cfg=aug_cfg)

    train_dataset = PlantDiseaseDataset(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = PlantDiseaseDataset(
        os.path.join(data_dir, "val"), transform=val_transform
    )

    # Print dataset statistics
    print(f"Dataset: {data_dir}")
    print(f"  Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    train_dist = train_dataset.get_class_distribution()
    print(f"  Train distribution: {train_dist}")

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)
    pin_memory = cfg["data"].get("pin_memory", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "num_classes": len(train_dataset.classes),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
    }
