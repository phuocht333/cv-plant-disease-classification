"""GhostNetV3 model wrapper.

TODO: GhostNetV3 may not be available in timm yet. Options:
  1. Use official GhostNetV3 repository implementation
  2. Use timm's GhostNet v1/v2 as fallback: timm.create_model('ghostnetv2_100')
  3. Manually integrate from: https://github.com/huawei-noah/Efficient-AI-Backbones
"""

import timm
import torch.nn as nn


def create_ghostnetv3(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create GhostNetV3 model.

    Falls back to GhostNet v2 if v3 is not available in timm.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        GhostNet model with custom classifier head.
    """
    # Try GhostNetV3 first, fall back to V2 if not available
    try:
        model = timm.create_model(
            "ghostnetv3_100",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except RuntimeError:
        model = timm.create_model(
            "ghostnetv2_100",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    return model
