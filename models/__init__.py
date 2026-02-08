"""Model factory for plant disease classification architectures."""

from models.mobilenetv4 import create_mobilenetv4
from models.convnextv2 import create_convnextv2
from models.ghostnetv3 import create_ghostnetv3


_MODEL_REGISTRY = {
    "mobilenetv4_conv_small": create_mobilenetv4,
    "convnextv2_nano": create_convnextv2,
    "ghostnetv3": create_ghostnetv3,
}


def get_model(name: str, num_classes: int, pretrained: bool = True):
    """Create a model by name.

    Args:
        name: Model identifier (e.g. 'mobilenetv4_conv_small', 'convnextv2_nano', 'ghostnetv3').
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        nn.Module: The instantiated model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)
