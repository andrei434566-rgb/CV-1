"""Train independent one-class YOLO segmentation for whole core columns.

Requires local labeled CoreBreakout data and a CUDA-capable machine.
This does not alter the CV-1 92-class facies model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_dataset(folder: Path) -> dict:
    manifest = json.loads((folder / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest["task"] != "core-column-instance-segmentation" or manifest["class_names"] != ["core_column"]:
        raise ValueError("Expected a 1-class core-column segmentation dataset")
    images, groups = {"train": set(), "val": set()}, {"train": set(), "val": set()}
    for row in manifest["samples"]:
        split = row["split"]
        if split not in images:
            raise ValueError(f"Invalid split: {split}")
        images[split].add(row["source_sha256"])
        groups[split].add(row["well_group"] or row["source_sha256"])
        if not (folder / row["image"]).is_file() or not (folder / row["labels"]).is_file():
            raise ValueError(f"Missing image or label: {row['image']}")
    if not all(images.values()) or images["train"] & images["val"] or groups["train"] & groups["val"]:
        raise ValueError("Empty split or source/well leakage")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained YOLO segmentation weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="CUDA device index; deliberately use cpu to override")
    parser.add_argument("--out", type=Path, default=Path("runs/core_columns"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("epochs and batch must be positive")
    manifest = validate_dataset(args.dataset)
    if manifest["split_strategy"] != "well":
        print("WARNING: photo-only val; results do NOT measure generalization to unseen wells.")
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Use --device cpu only for a deliberately slow test.")
    from ultralytics import YOLO
    model = YOLO(args.model)
    if model.task != "segment":
        raise ValueError("Use segmentation weights, e.g. yolo11n-seg.pt")
    result = model.train(
        data=str((args.dataset / "data.yaml").resolve()),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, patience=15, seed=42, deterministic=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, flipud=0.0,
        degrees=0.0, perspective=0.0,
        project=str(args.out.resolve()), name="core_columns", exist_ok=False)
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training did not produce {best}")
    print(f"Ready: {best}")
    print("One-class column model; do NOT replace the main facies YOLO weights.")


if __name__ == "__main__":
    main()
