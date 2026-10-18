"""Stage 32k unused source-labeled Mendeley crops for a 40k CV experiment.

Training is limited to the 20 wells in the existing development split. Images
from validation/test wells and prior experiment manifests are excluded. This is
small-crop lithology classification only, not full core descriptions or facies.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/external_raw/mendeley_lithology"
DEST = ROOT / "data/mendeley_lithology_40000"
SOURCE_INDEX = RAW / "source_index.jsonl"
ARCHIVE = RAW / "b6rj29mrss-1.zip"
PRIOR = ROOT / "data/mendeley_lithology_pretrained_holdout/summary.json"
BASE_CARD = ROOT / "model_spec_mendeley_resnet18_imagenet_8000.json"
CLASSES = ("sandstone", "limestone", "shale", "garbage")
ADDITIONAL_PER_CLASS = 8000
SEED = 20261013


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    card = json.loads(BASE_CARD.read_text(encoding="utf-8"))
    train_wells = set(card["training"]["data_audit"]["train_wells"])
    val_wells = set(card["validation"]["wells"])
    test_wells = set(card["test"]["wells"])
    if train_wells & (val_wells | test_wells) or val_wells & test_wells:
        raise ValueError("Wells leak across splits")

    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    used_members: set[str] = set()
    used_hashes: set[str] = set()
    for manifest in prior["excluded_prior_manifests"]:
        for row in rows(Path(manifest)):
            meta = row["metadata"]
            used_members.add(meta["source_archive_member"])
            used_hashes.add(meta["image_sha256"])
    for manifest in (ROOT / "data/mendeley_lithology_pretrained_holdout/val.jsonl",
                     ROOT / "data/mendeley_lithology_pretrained_holdout/test.jsonl"):
        for row in rows(manifest):
            used_members.add(row["metadata"]["source_archive_member"])
            used_hashes.add(row["metadata"]["image_sha256"])

    source = rows(SOURCE_INDEX)
    candidates = {name: [] for name in CLASSES}
    for row in source:
        if (row["source_well_id"] in train_wells
                and row["source_split"] == "train"
                and row["source_label"] in CLASSES
                and row["eligible_for_lithology_pilot"] is True
                and row["archive_member"] not in used_members):
            candidates[row["source_label"]].append(row)
    randomizer = random.Random(SEED)
    for group in candidates.values():
        randomizer.shuffle(group)
        if len(group) < ADDITIONAL_PER_CLASS:
            raise ValueError("Too few unused source images")

    output_rows = []
    counts: Counter[str] = Counter()
    skipped_bad = 0
    skipped_hash = 0
    DEST.joinpath("images").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for source_class in CLASSES:
            for row in candidates[source_class]:
                if counts[source_class] >= ADDITIONAL_PER_CLASS:
                    break
                name = row["archive_member"]
                image_bytes = archive.read(name)
                digest = hashlib.sha256(image_bytes).hexdigest()
                if digest in used_hashes:
                    skipped_hash += 1
                    continue
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        dimensions = list(image.size)
                except Exception:
                    skipped_bad += 1
                    continue
                photo = DEST / "images" / f"{source_class}_{digest[:20]}.jpeg"
                photo.write_bytes(image_bytes)
                output_rows.append({
                    "image": str(photo.resolve()),
                    "metadata": {
                        "source_archive_member": name,
                        "source_label_en": source_class,
                        "well_id": row["source_well_id"],
                        "image_sha256": digest,
                        "image_dimensions_px": dimensions,
                        "label_provenance": "source_direct_folder",
                        "training_approved": True,
                        "geology_approved": False,
                        "usage_scope": "small_crop_lithology_only",
                    },
                })
                used_hashes.add(digest)
                counts[source_class] += 1
            if counts[source_class] != ADDITIONAL_PER_CLASS:
                raise ValueError(f"Insufficient unique images for {source_class}: {counts[source_class]}")

    manifest = DEST / "additional_train.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "source": "https://data.mendeley.com/datasets/b6rj29mrss/1",
        "license": "CC BY 4.0",
        "seed": SEED,
        "additional_count": len(output_rows),
        "additional_by_class": dict(counts),
        "original_train_wells": sorted(train_wells),
        "validation_wells_excluded": sorted(val_wells),
        "test_wells_excluded": sorted(test_wells),
        "source_original_split": "train only for additional images",
        "excluded_prior_source_members": len(used_members),
        "skipped_corrupt_images": skipped_bad,
        "skipped_duplicate_hashes": skipped_hash,
        "existing_train_count": 8000,
        "combined_train_count": 8000 + len(output_rows),
        "training_scope": "experimental Mendeley four-class 2x2 cm crop classification; no descriptions or facies",
        "manifest": str(manifest.relative_to(ROOT)),
    }
    (DEST / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
