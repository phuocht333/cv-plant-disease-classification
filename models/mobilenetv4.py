"""MobileNetV4 model wrapper using timm."""

import timm
import torch.nn as nn


def create_mobilenetv4(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV4-Conv-Small model.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        MobileNetV4 model with custom classifier head.
    """
    model = timm.create_model(
        "mobilenetv4_conv_small",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
