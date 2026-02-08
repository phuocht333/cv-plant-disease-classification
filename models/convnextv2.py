"""ConvNeXt-V2 Nano model wrapper using timm."""

import timm
import torch.nn as nn


def create_convnextv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ConvNeXt-V2 Nano model.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        ConvNeXt-V2 Nano model with custom classifier head.
    """
    model = timm.create_model(
        "convnextv2_nano",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
