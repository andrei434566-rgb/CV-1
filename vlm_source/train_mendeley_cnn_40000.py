"""Continue the four-class Mendeley crop classifier on 40k source labels.

This controlled experiment trains on whole-well-disjoint crops and selects its
checkpoint on validation wells only. It does not train a description VLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from train_mendeley_cnn import CLASSES, CoreDataset, evaluate


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/core_cv_mendeley_resnet18_imagenet_8000/model.pt"
TRAIN = (
    ROOT / "data/mendeley_lithology_4000/train.jsonl",
    ROOT / "data/mendeley_lithology_next/train.jsonl",
    ROOT / "data/mendeley_lithology_40000/additional_train.jsonl",
)
VAL = ROOT / "data/mendeley_lithology_pretrained_holdout/val.jsonl"
TEST = ROOT / "data/mendeley_lithology_pretrained_holdout/test.jsonl"
DEST = ROOT / "outputs/core_cv_mendeley_resnet18_imagenet_40000"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(train: list[dict], val: list[dict], test: list[dict]) -> dict:
    if (len(train), len(val), len(test)) != (40000, 400, 400):
        raise ValueError("Unexpected split sizes")
    split_wells = [{row["metadata"]["well_id"] for row in part} for part in (train, val, test)]
    if any(split_wells[a] & split_wells[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Wells overlap across train/validation/test")
    split_hashes = []
    for part in (train, val, test):
        hashes = [row["metadata"]["image_sha256"] for row in part]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Duplicate image hashes in split")
        split_hashes.append(set(hashes))
        for row in part:
            meta = row["metadata"]
            if meta["source_label_en"] not in CLASSES or meta["label_provenance"] != "source_direct_folder":
                raise ValueError("Unverified source class")
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
    if any(split_hashes[a] & split_hashes[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Image content leak")
    if not all(row["metadata"]["training_approved"] is True for row in train):
        raise ValueError("Training record without permitted scope")
    return {
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "wells": {name: sorted(group) for name, group in zip(("train", "val", "test"), split_wells)},
        "train_by_class": dict(Counter(row["metadata"]["source_label_en"] for row in train)),
        "sha256_unique_per_split": [len(group) for group in split_hashes],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEST)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    random.seed(20261013)
    np.random.seed(20261013)
    torch.manual_seed(20261013)
    torch.cuda.manual_seed_all(20261013)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    train = [row for path in TRAIN for row in rows(path)]
    val = rows(VAL)
    test = rows(TEST)
    data_audit = audit(train, val, test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(data_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    train_loader = DataLoader(CoreDataset(train, True, 224, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(CoreDataset(val, False, 224, True), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(BASE, map_location="cpu", weights_only=True))
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(images)
        scheduler.step()
        measured = evaluate(model, val_loader)
        item = {"epoch": epoch, "train_loss": total_loss / len(train),
                "validation": measured, "elapsed_seconds": round(time.monotonic() - started, 1)}
        history.append(item)
        if measured["macro_f1"] > best_f1:
            best_f1, best_epoch = measured["macro_f1"], epoch
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                       args.output_dir / "model.pt")
        (args.output_dir / "training_history.json").write_text(
            json.dumps({"data_audit": data_audit, "epochs": history,
                        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
                        "settings": vars(args), "initial_checkpoint": str(BASE.relative_to(ROOT)),
                        "initial_checkpoint_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
                        "test_evaluated_during_training": False,
                        "scope": "four-class small-crop lithology only"},
                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"epoch": epoch, "train_loss": round(item["train_loss"], 4),
                          "val_correct": measured["correct"], "val_macro_f1": round(measured["macro_f1"], 4),
                          "best_epoch": best_epoch, "elapsed_seconds": item["elapsed_seconds"]},
                         ensure_ascii=False), flush=True)
        if epoch - best_epoch >= args.patience:
            break


if __name__ == "__main__":
    main()
