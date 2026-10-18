"""Evaluate the selected specialist CNN on source-labeled whole-well crops."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from train_mendeley_cnn import CLASSES, DEST, ROOT, TRAIN_MANIFESTS, CoreDataset, read


def score(rows: list[dict]) -> dict:
    classes = {}
    f1s = []
    for label in CLASSES:
        tp = sum(row["expected_en"] == label and row["predicted_en"] == label for row in rows)
        fp = sum(row["expected_en"] != label and row["predicted_en"] == label for row in rows)
        fn = sum(row["expected_en"] == label and row["predicted_en"] != label for row in rows)
        classes[label] = {"correct": tp, "total": tp + fn,
                          "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / (tp + fn) if tp + fn else None,
                          "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0}
        f1s.append(classes[label]["f1"])
    return {"count": len(rows), "correct": sum(row["expected_en"] == row["predicted_en"] for row in rows),
            "macro_f1": sum(f1s) / len(f1s), "per_class": classes,
            "prediction_distribution": dict(Counter(row["predicted_en"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEST)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mendeley_lithology_next")
    args = parser.parse_args()
    torch.set_num_threads(4)
    train = [row for path in TRAIN_MANIFESTS for row in read(path)]
    heldout = read(args.data_dir / f"{args.split}.jsonl")
    train_wells = {row["metadata"]["well_id"] for row in train}
    heldout_wells = {row["metadata"]["well_id"] for row in heldout}
    if train_wells & heldout_wells or len(heldout) != 400 or len(heldout_wells) != 4:
        raise ValueError("Unexpected whole-well holdout")
    train_hashes = {row["metadata"]["image_sha256"] for row in train}
    heldout_hashes = {row["metadata"]["image_sha256"] for row in heldout}
    if train_hashes & heldout_hashes or len(heldout_hashes) != len(heldout):
        raise ValueError("Image content overlap")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.to("cuda")
    model.eval()
    history = json.loads((args.model_dir / "training_history.json").read_text(encoding="utf-8"))
    loader = DataLoader(CoreDataset(heldout, False, history["image_side_px"],
                                    history.get("imagenet_normalize", False)), batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=True)
    records = []
    index = 0
    with torch.inference_mode():
        for images, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images.to("cuda", non_blocking=True))
            probs = logits.float().softmax(dim=1)
            values, predictions = probs.max(dim=1)
            for prediction, confidence in zip(predictions.cpu().tolist(), values.cpu().tolist()):
                source = heldout[index]
                meta = source["metadata"]
                records.append({"record_id": meta["record_id"], "well_id": meta["well_id"],
                                "image": source["image"], "source_archive_member": meta["source_archive_member"],
                                "expected_en": meta["source_label_en"],
                                "predicted_en": CLASSES[prediction],
                                "model_softmax_max_uncalibrated": confidence})
                index += 1
    by_well = defaultdict(list)
    for record in records:
        by_well[record["well_id"]].append(record)
    report = {
        "scope": "Four source-folder lithology/non-core labels on 2x2cm crops; no descriptions or facies",
        "split": args.split, "model": str((args.model_dir / "model.pt").resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "selected_epoch": history["best_epoch"],
        "train_wells": sorted(train_wells), "heldout_wells": sorted(heldout_wells),
        "train_images": len(train), "heldout_images": len(heldout),
        "score": score(records),
        "per_well": {well: score(rows) for well, rows in sorted(by_well.items())},
        "limitations": ["Source folder labels are not individually reviewed by a geologist.",
                        "This is a specialist CV classifier, not the local VLM description model.",
                        "The same well groups appeared in earlier research runs, although these 400 images were not used before.",
                        "Softmax confidence is not calibrated; do not interpret it as geological certainty."],
        "records": records,
    }
    output = args.output or args.model_dir / f"{args.split}_evaluation.json"
    if output.exists():
        raise FileExistsError(f"Preserving earlier evaluation: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "score": report["score"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
