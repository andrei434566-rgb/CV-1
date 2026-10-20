"""Prepare a group-safe YOLO core-column segmentation dataset from CoreBreakout LabelMe files.

This is a separate ONE-CLASS model. Do not merge it into CV-1's 92 facies classes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def is_uv(name: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])uv(?![a-z0-9])|ультрафиолет", name.casefold()))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_groups(csv_path: Path | None) -> dict[str, str]:
    if csv_path is None:
        return {}
    result = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"image", "well_id"}.issubset(rows.fieldnames or []):
            raise ValueError("CSV must have columns image,well_id")
        for row in rows:
            name = Path((row["image"] or "").strip()).name.casefold()
            well = (row["well_id"] or "").strip().casefold()
            if not name or not well or (name in result and result[name] != well):
                raise ValueError(f"Missing or conflicting group for {name}")
            result[name] = well
    if not result:
        raise ValueError("Well mapping is empty")
    return result


def collect(source: Path, well_map: dict[str, str], allow_photo: bool) -> tuple[list[dict], list[str]]:
    if not source.is_dir():
        raise NotADirectoryError(source)
    records, skipped, seen = [], [], {}
    for ann in sorted(source.rglob("*.json")):
        if is_uv(ann.stem):
            skipped.append(f"UV: {ann}")
            continue
        try:
            obj = json.loads(ann.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError):
            skipped.append(f"Not LabelMe: {ann}")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("shapes"), list):
            skipped.append(f"Not LabelMe: {ann}")
            continue
        names = [ann.with_suffix(e) for e in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")]
        if obj.get("imagePath"):
            names.append(ann.parent / Path(str(obj["imagePath"])).name)
        image = next((p for p in names if p.is_file()), None)
        if image is None:
            raise FileNotFoundError(f"No image for {ann}")
        if is_uv(image.name):
            skipped.append(f"UV: {image}")
            continue
        w, h = int(obj.get("imageWidth") or 0), int(obj.get("imageHeight") or 0)
        if w < 10 or h < 10:
            raise ValueError(f"Bad dimensions: {ann}")
        lines = []
        for shape in obj["shapes"]:
            if not str(shape.get("label") or "").casefold().startswith("col"):
                continue  # 'tray' is not a core column
            points = shape.get("points") or []
            if len(points) < 3:
                raise ValueError(f"Bad column polygon: {ann}")
            coords = []
            for px, py in points:
                x, y = float(px), float(py)
                if not (0 <= x <= w and 0 <= y <= h):
                    raise ValueError(f"Polygon outside image: {ann}")
                coords += [f"{x/w:.6f}", f"{y/h:.6f}"]
            lines.append("0 " + " ".join(coords))
        well = well_map.get(image.name.casefold(), "")
        if well_map and not well:
            raise ValueError(f"Missing well mapping for {image.name}")
        digest = file_sha256(image)
        if digest in seen:
            previous = seen[digest]
            if previous["well"] != well or previous["lines"] != lines:
                raise ValueError(f"Duplicate photo with conflicting labels/well: {image}")
            skipped.append(f"Duplicate: {image}")
            continue
        item = {"image": image, "well": well, "hash": digest, "lines": lines}
        seen[digest] = item
        records.append(item)
    if not records or len(records) < 2 or not any(x["lines"] for x in records):
        raise ValueError("Need at least 2 photos and nonempty core-column polygons")
    if not well_map and not allow_photo:
        raise ValueError("Provide --groups-csv for independent wells; --photo-holdout is research-only")
    return records, skipped


def prepare(source: Path, destination: Path, groups_csv: Path | None = None,
            allow_photo_holdout: bool = False, folds: int = 5) -> dict:
    if destination.exists():
        raise FileExistsError(f"Cannot overwrite {destination}")
    mapping = load_groups(groups_csv)
    records, skipped = collect(source, mapping, allow_photo_holdout)
    from sklearn.model_selection import GroupKFold
    groups = [x["well"] or x["hash"] for x in records]
    n = min(folds, len(set(groups)))
    if n < 2:
        raise ValueError("Need at least two independent groups")
    best = None
    for fold, (train, val) in enumerate(GroupKFold(n_splits=n).split(records, groups=groups)):
        if not any(records[i]["lines"] for i in train) or not any(records[i]["lines"] for i in val):
            continue
        rank = abs(len(val)/len(records) - 1/n), fold
        if best is None or rank < best[0]:
            best = rank, set(val), fold
    if best is None:
        raise ValueError("No split keeps positive column labels in train and val")
    _, val, fold = best
    destination.mkdir(parents=True)
    rows = []
    for i, item in enumerate(records):
        split = "val" if i in val else "train"
        images = destination / "images" / split
        labels = destination / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        name = f"core_{i + 1:05d}_{item['hash'][:12]}"
        image = images / (name + item["image"].suffix.lower())
        label = labels / (name + ".txt")
        shutil.copy2(item["image"], image)
        label.write_text("\n".join(item["lines"]) + ("\n" if item["lines"] else ""), encoding="utf-8")
        rows.append({"split": split, "source_sha256": item["hash"], "well_group": item["well"] or None,
                     "image": image.relative_to(destination).as_posix(),
                     "labels": label.relative_to(destination).as_posix(), "objects": len(item["lines"])})
    train_groups = {groups[i] for i in range(len(records)) if i not in val}
    val_groups = {groups[i] for i in val}
    if train_groups & val_groups:
        raise AssertionError("Train/val group leakage")
    (destination / "data.yaml").write_text(
        f"path: {json.dumps(destination.resolve().as_posix(), ensure_ascii=False)}\n"
        "train: images/train\nval: images/val\nnames:\n  0: core_column\n", encoding="utf-8")
    report = {"task": "core-column-instance-segmentation", "class_names": ["core_column"],
              "source": "CoreBreakout LabelMe", "split_method": "GroupKFold", "fold_count": n,
              "selected_fold": fold, "split_strategy": "well" if groups_csv else "photo_only",
              "train_groups": len(train_groups), "val_groups": len(val_groups),
              "total_images": len(rows), "total_columns": sum(x["objects"] for x in rows),
              "counts": dict(Counter(x["split"] for x in rows)), "skipped": skipped, "samples": rows,
              "warning": "Research-only until source-data permissions are verified. NOT a 92-class facies model."}
    (destination / "dataset_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Unpacked CoreBreakout assets/data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path, help="CSV with image,well_id")
    parser.add_argument("--photo-holdout", action="store_true", help="Research only, no unseen-well accuracy claims")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    report = prepare(args.input, args.output, groups_csv=args.groups_csv,
                     allow_photo_holdout=args.photo_holdout, folds=args.folds)
    print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "skipped")},
                     ensure_ascii=False, indent=2))
