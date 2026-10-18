"""Evaluate the selected EfficientNet on fixed Mendeley validation/test wells."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from evaluate_mendeley_cnn import score
from train_mendeley_cnn import CLASSES, CoreDataset
from train_mendeley_cnn_40000 import TEST, TRAIN, VAL, rows
from train_mendeley_efficientnet_40000 import DEST


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    train = [row for manifest in TRAIN for row in rows(manifest)]
    heldout = rows(VAL if args.split == "val" else TEST)
    train_wells = {row["metadata"]["well_id"] for row in train}
    heldout_wells = {row["metadata"]["well_id"] for row in heldout}
    train_hashes = {row["metadata"]["image_sha256"] for row in train}
    if (len(train) != 40000 or len(heldout) != 400 or train_wells & heldout_wells
            or train_hashes & {row["metadata"]["image_sha256"] for row in heldout}):
        raise ValueError("Holdout identity/content audit failed")
    history = json.loads((DEST / "training_history.json").read_text(encoding="utf-8"))
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    checkpoint = DEST / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to("cuda").eval()
    loader = DataLoader(CoreDataset(heldout, False, 224, True), batch_size=64,
                        shuffle=False, num_workers=0, pin_memory=True)
    records = []
    index = 0
    with torch.inference_mode():
        for images, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images.to("cuda", non_blocking=True))
            probabilities = logits.float().softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)
            for prediction, probability in zip(predictions.cpu().tolist(), confidence.cpu().tolist()):
                row = heldout[index]
                meta = row["metadata"]
                records.append({
                    "record_id": meta["record_id"],
                    "well_id": meta["well_id"],
                    "image": row["image"],
                    "source_archive_member": meta["source_archive_member"],
                    "expected_en": meta["source_label_en"],
                    "predicted_en": CLASSES[prediction],
                    "model_softmax_max_uncalibrated": probability,
                })
                index += 1
    by_well = defaultdict(list)
    for record in records:
        by_well[record["well_id"]].append(record)
    result = {
        "scope": "Mendeley 2x2 cm four-class crop lithology/non-core; not full description or facies",
        "split": args.split,
        "selected_epoch": history["best_epoch"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_count": len(train),
        "heldout_count": len(heldout),
        "train_wells": sorted(train_wells),
        "heldout_wells": sorted(heldout_wells),
        "score": score(records),
        "per_well": {well: score(group) for well, group in sorted(by_well.items())},
        "records": records,
        "limitations": [
            "The same heldout well groups appeared in earlier project research.",
            "Source folder labels were not individually reviewed by a geologist.",
            "Crops from a well are correlated; image-level accuracy is optimistic as an uncertainty estimate.",
            "This classifier has not been tested on local ancient core and cannot generate Russian descriptions or facies.",
        ],
    }
    output = DEST / f"{args.split}_evaluation.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"split": args.split, "score": result["score"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
