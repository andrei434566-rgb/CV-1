"""Local four-class inference for a single small crop of a core photograph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from train_mendeley_cnn import CLASSES, ROOT, RU


DEFAULT_MODEL = ROOT / "outputs/core_cv_mendeley_resnet18_imagenet_8000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize only the Mendeley-style 2x2cm crop class")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    history = json.loads((args.model_dir / "training_history.json").read_text(encoding="utf-8"))
    if not (args.model_dir / "model.pt").is_file():
        raise FileNotFoundError(args.model_dir / "model.pt")
    side = history["image_side_px"]
    if history.get("imagenet_normalize"):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,) * 3, (0.5,) * 3
    transform = transforms.Compose([transforms.Resize((side, side)), transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    with Image.open(args.image) as source:
        original_size = list(source.size)
        image = transform(source.convert("RGB")).unsqueeze(0)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    with torch.inference_mode():
        probabilities = model(image.to(device)).float().softmax(dim=1)[0].cpu().tolist()
    winner = max(range(len(probabilities)), key=probabilities.__getitem__)
    print(json.dumps({
        "scope": "Mendeley-like small-crop lithology/non-core only; no full core description or facies",
        "image": str(args.image.resolve()), "input_dimensions_px": original_size,
        "predicted_source_class": CLASSES[winner], "predicted_ru": RU[winner],
        "softmax_scores_uncalibrated": {RU[i]: probabilities[i] for i in range(len(CLASSES))},
        "model": str((args.model_dir / "model.pt").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
