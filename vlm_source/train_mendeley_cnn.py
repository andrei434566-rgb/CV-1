"""Train a compact specialist photo classifier on source-labeled Mendeley crops.

This is a separate CV baseline for Kern Analyzer, not a VLM description model.
Whole-well validation is used for checkpoint selection; no test rows are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


ROOT = Path(__file__).resolve().parent
CLASSES = ("sandstone", "limestone", "shale", "garbage")
RU = ("песчаник", "известняк", "глинистая порода", "не керн")
TRAIN_MANIFESTS = (
    ROOT / "data/mendeley_lithology_4000/train.jsonl",
    ROOT / "data/mendeley_lithology_next/train.jsonl",
)
VAL_MANIFEST = ROOT / "data/mendeley_lithology_next/val.jsonl"
DEST = ROOT / "outputs/core_cv_mendeley_resnet18_8000"
SIDE = 160


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(train: list[dict], val: list[dict]) -> dict:
    if len(train) != 8000 or len(val) != 400:
        raise ValueError("Unexpected training or validation sample count")
    for row in train + val:
        meta = row["metadata"]
        if meta["label_provenance"] != "source_direct_folder" or meta["source_label_en"] not in CLASSES:
            raise ValueError("Non-source lithology label")
        if row["conversations"][1]["value"] != RU[CLASSES.index(meta["source_label_en"])]:
            raise ValueError("Russian target does not match source class")
        if not Path(row["image"]).is_file():
            raise FileNotFoundError(row["image"])
    if not all(row["metadata"]["training_approved"] is True for row in train):
        raise ValueError("Training permission flag not set")
    train_wells = {row["metadata"]["well_id"] for row in train}
    val_wells = {row["metadata"]["well_id"] for row in val}
    if train_wells & val_wells or len(train_wells) != 20 or len(val_wells) != 4:
        raise ValueError("Whole-well separation failed")
    train_hashes = [row["metadata"]["image_sha256"] for row in train]
    val_hashes = [row["metadata"]["image_sha256"] for row in val]
    if len(set(train_hashes)) != len(train_hashes) or set(train_hashes) & set(val_hashes):
        raise ValueError("Image content leak in data split")
    return {"train_wells": sorted(train_wells), "val_wells": sorted(val_wells),
            "train_by_source_class": dict(Counter(row["metadata"]["source_label_en"] for row in train)),
            "val_by_source_class": dict(Counter(row["metadata"]["source_label_en"] for row in val)),
            "train_unique_hashes": len(set(train_hashes)), "val_unique_hashes": len(set(val_hashes))}


class CoreDataset(Dataset):
    def __init__(self, rows: list[dict], training: bool, image_side: int = SIDE,
                 imagenet_normalize: bool = False):
        self.rows = rows
        augment = [transforms.Resize((image_side, image_side))]
        if training:
            augment += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05)]
        if imagenet_normalize:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean, std = (0.5,) * 3, (0.5,) * 3
        augment += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        self.transform = transforms.Compose(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASSES.index(row["metadata"]["source_label_en"])


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    expected = []
    predicted = []
    for images, targets in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images.to("cuda", non_blocking=True))
        expected.extend(targets.tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
    per_class = {}
    f1s = []
    for code, name in enumerate(CLASSES):
        tp = sum(a == code and b == code for a, b in zip(expected, predicted))
        fp = sum(a != code and b == code for a, b in zip(expected, predicted))
        fn = sum(a == code and b != code for a, b in zip(expected, predicted))
        per_class[name] = {"correct": tp, "total": tp + fn,
                           "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0}
        f1s.append(per_class[name]["f1"])
    return {"correct": sum(a == b for a, b in zip(expected, predicted)),
            "total": len(expected), "macro_f1": sum(f1s) / len(f1s),
            "per_class": per_class,
            "prediction_distribution": dict(Counter(CLASSES[p] for p in predicted))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0,
                        help="Data loader workers; zero avoids Windows spawn stalls")
    parser.add_argument("--pretrained-weights", type=Path,
                        help="Official torchvision ResNet18 ImageNet state dict (local file)")
    parser.add_argument("--val-manifest", type=Path, default=VAL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEST)
    parser.add_argument("--image-side", type=int, default=SIDE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    random.seed(20261012)
    np.random.seed(20261012)
    torch.manual_seed(20261012)
    torch.cuda.manual_seed_all(20261012)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    train = [row for path in TRAIN_MANIFESTS for row in read(path)]
    val = read(args.val_manifest)
    data_audit = audit(train, val)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(data_audit, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    imagenet_normalize = args.pretrained_weights is not None
    train_loader = DataLoader(CoreDataset(train, True, args.image_side, imagenet_normalize),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(CoreDataset(val, False, args.image_side, imagenet_normalize),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    model = resnet18(weights=None)
    pretrained_sha256 = None
    if args.pretrained_weights:
        pretrained_sha256 = hashlib.sha256(args.pretrained_weights.read_bytes()).hexdigest()
        if args.pretrained_weights.name == "resnet18-f37072fd.pth" and not pretrained_sha256.startswith("f37072fd"):
            raise ValueError("Official pretrained weights SHA-256 prefix mismatch")
        model.load_state_dict(torch.load(args.pretrained_weights, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(images)
        scheduler.step()
        measured = evaluate(model, val_loader)
        item = {"epoch": epoch, "train_loss": total_loss / len(train),
                "learning_rate": optimizer.param_groups[0]["lr"], "validation": measured}
        history.append(item)
        if measured["macro_f1"] > best_f1:
            best_f1 = measured["macro_f1"]
            best_epoch = epoch
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, args.output_dir / "model.pt")
        (args.output_dir / "training_history.json").write_text(
            json.dumps({"data_audit": data_audit, "epochs": history, "best_epoch": best_epoch,
                        "best_macro_f1": best_f1, "best_validation": history[best_epoch - 1]["validation"],
                        "hyperparameters": {k: str(v) if isinstance(v, Path) else v
                                             for k, v in vars(args).items()},
                        "architecture": "torchvision.resnet18_imagenet1k_v1_finetuned"
                                        if imagenet_normalize else "torchvision.resnet18_random_init",
                        "pretrained_weights_sha256": pretrained_sha256,
                        "image_side_px": args.image_side,
                        "imagenet_normalize": imagenet_normalize,
                        "validation_manifest": str(args.val_manifest.resolve()),
                        "test_images_evaluated": 0,
                        "scope": "four-class Mendeley small-crop classification, not geological description/facies"},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"epoch": epoch, "loss": round(item["train_loss"], 4),
                          "val_correct": measured["correct"], "val_macro_f1": round(best_f1, 4),
                          "best_epoch": best_epoch, "per_class": measured["per_class"]},
                         ensure_ascii=False), flush=True)
        if epoch - best_epoch >= args.patience:
            break


if __name__ == "__main__":
    main()
