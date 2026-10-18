"""Stage a small, source-labeled lithology pilot from whole-well Mendeley splits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/external_raw/mendeley_lithology"
ARCHIVE = SOURCE / "b6rj29mrss-1.zip"
DEST = ROOT / "data/mendeley_lithology_pilot"
TARGETS = {"train": 120, "val": 40, "test": 50}
LABEL_RU = {
    "sandstone": "песчаник",
    "limestone": "известняк",
    "shale": "глинистая порода",
    "garbage": "не керн",
}
PROMPT = (
    "<image>\nОпредели один класс на маленьком фрагменте фотографии керна. "
    "Выбери ровно один вариант: песчаник, известняк, глинистая порода, не керн. "
    "Ответь только названием класса."
)


def choose_round_robin(items: list[dict], count: int, seed: int) -> list[dict]:
    by_well = defaultdict(list)
    for item in items:
        by_well[item["source_well_id"]].append(item)
    randomizer = random.Random(seed)
    queues = {}
    for well, values in sorted(by_well.items()):
        randomizer.shuffle(values)
        queues[well] = deque(values)
    wells = list(queues)
    randomizer.shuffle(wells)
    chosen = []
    while len(chosen) < count and wells:
        for well in wells[:]:
            if queues[well]:
                chosen.append(queues[well].popleft())
                if len(chosen) == count:
                    break
            if not queues[well]:
                wells.remove(well)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEST)
    parser.add_argument("--train-per-class", type=int, default=TARGETS["train"])
    parser.add_argument("--val-per-class", type=int, default=TARGETS["val"])
    parser.add_argument("--test-per-class", type=int, default=TARGETS["test"])
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="Skip source members and image hashes already used in an earlier staged manifest")
    parser.add_argument("--sample-pool-multiplier", type=int, default=2,
                        help="Number of round-robin source candidates to sample per target row")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    targets = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    if any(value < 0 for value in targets.values()):
        raise ValueError("Количество фрагментов не может быть отрицательным")
    if args.sample_pool_multiplier < 2:
        raise ValueError("sample-pool-multiplier must be at least 2")
    if destination.exists():
        raise FileExistsError("Пилот уже существует; не перезаписываю разметку: " + str(destination))
    excluded_members = set()
    excluded_hashes = set()
    for manifest in args.exclude_manifest:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            meta = json.loads(line)["metadata"]
            if meta.get("source_archive_member"):
                excluded_members.add(meta["source_archive_member"])
            if meta.get("image_sha256"):
                excluded_hashes.add(meta["image_sha256"])
    records = [json.loads(line) for line in (SOURCE / "source_index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    grouped = defaultdict(list)
    for item in records:
        if item["eligible_for_lithology_pilot"]:
            grouped[(item["well_split"], item["source_label"])].append(item)
    selected = []
    for split, target_per_class in targets.items():
        for label in LABEL_RU:
            chosen = choose_round_robin(grouped[(split, label)],
                                        target_per_class * args.sample_pool_multiplier,
                                        20261012 + len(selected))
            selected.extend(chosen)
    destination.mkdir(parents=True)
    accepted = defaultdict(list)
    errors = []
    seen_content = set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for item in selected:
            split, label = item["well_split"], item["source_label"]
            if len(accepted[(split, label)]) >= targets[split]:
                continue
            member = item["archive_member"]
            if member in excluded_members:
                continue
            try:
                payload = archive.read(member)
                digest = hashlib.sha256(payload).hexdigest()
                if digest in seen_content or digest in excluded_hashes:
                    errors.append({"archive_member": member, "reason": "exact_duplicate_or_excluded_content"})
                    continue
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                with Image.open(io.BytesIO(payload)) as image:
                    if image.width < 32 or image.height < 32:
                        errors.append({"archive_member": member, "reason": "too_small"})
                        continue
                    image.load()
                    dimensions = [image.width, image.height]
            except (OSError, ValueError, UnidentifiedImageError, zipfile.BadZipFile) as exc:
                errors.append({"archive_member": member, "reason": f"decode_error: {type(exc).__name__}"})
                continue
            seen_content.add(digest)
            image_path = destination / "images" / split / f"{label}_{digest[:20]}.jpeg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(payload)
            accepted[(split, label)].append({
                "image": str(image_path),
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt", "value": LABEL_RU[label]},
                ],
                "metadata": {
                    "record_id": digest[:20],
                    "dataset": "mendeley_alzubaidi_lithology_v1",
                    "source_url": "https://data.mendeley.com/datasets/b6rj29mrss/1",
                    "source_archive_member": member,
                    "source_original_split": item["source_split"],
                    "source_label_en": label,
                    "well_id": item["source_well_id"],
                    "well_split": split,
                    "image_sha256": digest,
                    "image_dimensions_px": dimensions,
                    "label_provenance": "source_direct_folder",
                    "training_approved": split == "train",
                    "geology_approved": False,
                    "usage_scope": "small_crop_lithology_only",
                },
            })
    for split in targets:
        output = destination / f"{split}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for label in LABEL_RU:
                for row in accepted[(split, label)]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {f"{split}/{label}": len(accepted[(split, label)]) for split in targets for label in LABEL_RU}
    summary = {
        "source_inventory": str(SOURCE / "inventory.json"),
        "class_mapping_ru": LABEL_RU,
        "prompt": PROMPT,
        "counts": counts,
        "decode_or_duplicate_exclusions": errors,
        "excluded_prior_manifests": [str(path.resolve()) for path in args.exclude_manifest],
        "excluded_prior_source_members": len(excluded_members),
        "excluded_prior_image_hashes": len(excluded_hashes),
        "wells_by_split": {split: sorted({item["metadata"]["well_id"] for label in LABEL_RU for item in accepted[(split, label)]}) for split in targets},
        "scope": "separate four-class lithology pilot; no descriptions or facies",
        "warning": "Source labels were not individually checked by a geologist. Model must remain experimental until held-out well evaluation and ancient-core transfer review.",
    }
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(counts[f"{split}/{label}"] == target for split, target in targets.items() for label in LABEL_RU), counts
    assert not (set(summary["wells_by_split"]["train"]) & set(summary["wells_by_split"]["val"]))
    assert not (set(summary["wells_by_split"]["train"]) & set(summary["wells_by_split"]["test"]))
    assert not (set(summary["wells_by_split"]["val"]) & set(summary["wells_by_split"]["test"]))
    print(json.dumps({"counts": counts, "excluded": len(errors), "wells_by_split": {k: len(v) for k, v in summary["wells_by_split"].items()}}, ensure_ascii=True))


if __name__ == "__main__":
    main()
