"""Single-image inference for plant disease classification.

Usage:
    python scripts/inference.py --model mobilenetv4_conv_small \
        --checkpoint outputs/checkpoints/mobilenetv4_conv_small_best.pth \
        --image /path/to/image.jpg \
        --class_names "Healthy,Leaf Blight,Powdery Mildew"
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import get_val_transforms
from models import get_model


def predict(model, image_path, transform, class_names, device):
    """Run inference on a single image.

    Args:
        model: Trained model in eval mode.
        image_path: Path to the input image.
        transform: Validation transform pipeline.
        class_names: List of class names.
        device: Device to run inference on.

    Returns:
        Tuple of (predicted_class_name, confidence, all_probabilities).
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    transformed = transform(image=image_np)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze()

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    pred_name = class_names[pred_idx] if class_names else str(pred_idx)

    return pred_name, confidence, probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Single-image plant disease inference")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--class_names", type=str, default=None,
                        help="Comma-separated class names")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse class names
    class_names = args.class_names.split(",") if args.class_names else None
    num_classes = len(class_names) if class_names else 10  # default fallback

    # Model
    model = get_model(args.model, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Transform
    transform = get_val_transforms(cfg["data"]["image_size"])

    # Predict
    pred_name, confidence, probs = predict(model, args.image, transform, class_names, device)

    print(f"\nImage: {args.image}")
    print(f"Prediction: {pred_name}")
    print(f"Confidence: {confidence:.4f}")

    if class_names:
        print("\nAll probabilities:")
        for name, prob in sorted(zip(class_names, probs), key=lambda x: -x[1]):
            print(f"  {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
