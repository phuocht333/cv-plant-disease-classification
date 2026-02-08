"""Custom PyTorch Dataset for greenhouse plant disease images."""

import os

from PIL import Image
from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    """Dataset class for plant disease classification.

    Expects a directory structure where each subdirectory represents a class:
        root/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
                ...

    Args:
        root_dir: Path to the dataset root directory.
        transform: Optional transform to apply to images.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (image_path, label)
        self.classes = []  # List of class names
        self.class_to_idx = {}  # Mapping from class name to index

        self._load_samples()

    def _load_samples(self):
        """Scan root_dir and build list of (path, label) pairs."""
        self.classes = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
