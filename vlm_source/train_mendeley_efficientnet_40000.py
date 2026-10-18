"""Train an independent four-class CV branch on licensed Mendeley core crops.

Test metadata is audited but test images are never loaded during training.
This cannot measure full-description VLM quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from train_mendeley_cnn import CLASSES, CoreDataset, evaluate
from train_mendeley_cnn_40000 import ROOT, TRAIN, VAL, TEST, audit, rows


PRETRAINED = ROOT / "model_cache/torchvision/efficientnet_b0_rwightman-7f5810bc.pth"
DEST = ROOT / "outputs/core_cv_mendeley_efficientnet_b0_40000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEST)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    weight_hash = hashlib.sha256(PRETRAINED.read_bytes()).hexdigest()
    if not weight_hash.startswith("7f5810bc"):
        raise ValueError("Official EfficientNet-B0 pretrained checksum mismatch")
    random.seed(20261014)
    np.random.seed(20261014)
    torch.manual_seed(20261014)
    torch.cuda.manual_seed_all(20261014)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    train = [row for manifest in TRAIN for row in rows(manifest)]
    val = rows(VAL)
    test = rows(TEST)  # audit identities and hashes only; never create test loader here
    data_audit = audit(train, val, test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(data_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    train_loader = DataLoader(CoreDataset(train, True, 224, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(CoreDataset(val, False, 224, True), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    model = efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(PRETRAINED, map_location="cpu", weights_only=True))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.to("cuda")
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 3e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    history = []
    best_epoch, best_f1 = 0, -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        loss_sum = 0.0
        for images, targets in train_loader:
            images = images.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(images)
        scheduler.step()
        measured = evaluate(model, val_loader)
        item = {"epoch": epoch, "train_loss": loss_sum / len(train), "validation": measured,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "feature_lr": optimizer.param_groups[0]["lr"],
                "classifier_lr": optimizer.param_groups[1]["lr"]}
        history.append(item)
        if measured["macro_f1"] > best_f1:
            best_epoch, best_f1 = epoch, measured["macro_f1"]
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, args.output_dir / "model.pt")
        report = {"data_audit": data_audit, "epochs": history, "best_epoch": best_epoch,
                  "best_validation_macro_f1": best_f1, "initial_weight_sha256": weight_hash,
                  "architecture": "torchvision.efficientnet_b0_imagenet1k_v1",
                  "settings": vars(args), "test_evaluated_during_training": False,
                  "scope": "Mendeley 2x2-cm four-class crop lithology, not full core description or facies"}
        (args.output_dir / "training_history.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"epoch": epoch, "train_loss": round(item["train_loss"], 4),
                          "val_correct": measured["correct"], "val_macro_f1": round(measured["macro_f1"], 4),
                          "best_epoch": best_epoch, "elapsed_seconds": item["elapsed_seconds"]}), flush=True)
        if epoch - best_epoch >= args.patience:
            break


if __name__ == "__main__":
    main()
