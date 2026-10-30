"""Pure-mask UniBo facies crops for a separate noncommercial VLM experiment.

Uses the existing prepared source-group split, never opens its reserved test
pixels, and reconstructs an exact-colour mask from the original annotation.
Publisher train/validation/test images were regrouped before this experiment;
the publisher's original Test folder is not a sealed test for this new split.
No labels or masks are drawn onto inputs. S-prefixes are group proxies only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://amsacta.unibo.it/id/eprint/7308/"
LICENSE = "CC BY-NC-SA 4.0"
CLASSES = (
    (1, "WDF", "Хорошо дренируемая пойма", (242, 229, 136)),
    (2, "PDF", "Плохо дренируемая пойма", (128, 193, 123)),
    (3, "P", "Продельта", (106, 69, 149)),
    (4, "PL", "Прослой торфа", (32, 75, 155)),
    (5, "FS", "Русловой песок", (242, 137, 21)),
    (6, "Sw", "Болото", (102, 179, 45)),
)
PROMPT = (
    "Определи исходный класс голоценовой осадочной фации на фрагменте керна "
    "из исследовательского набора UniBo. Варианты: "
    + "; ".join(item[2] for item in CLASSES)
    + '. Верни только JSON с одним строковым полем "facies_name" и точным названием одного класса. '
      "Не добавляй глубины, обстановки или описание."
)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_digest(image):
    value = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
    value.update(image.tobytes())
    return value.hexdigest()


def source_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Prepared manifest contains an unsafe path")
    return path


def exact_mask(rgb):
    """Only publisher canonical colours count; no tolerance rounding."""
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    result[np.all(rgb == (0, 0, 0), axis=2)] = 0
    for class_id, _, _, colour in CLASSES:
        result[np.all(rgb == colour, axis=2)] = class_id
    return result


def mask_quality(mask, class_id):
    area = mask.size
    return {
        "target_area_fraction": float((mask == class_id).sum() / area),
        "foreground_fraction": float(((mask >= 1) & (mask <= 6)).sum() / area),
        "ignore_fraction": float((mask == 255).sum() / area),
    }


def eligible_quality(quality):
    return (quality["target_area_fraction"] >= .98 and quality["foreground_fraction"] >= .98
            and quality["ignore_fraction"] <= .005)


def validate_source_groups(rows):
    group_splits, source_splits = defaultdict(set), defaultdict(set)
    for row in rows:  # metadata only; prepared S-group test pixels stay unopened
        group_splits[str(row["well_group"])].add(row["split"])
        source_splits[row["source_sha256"]].add(row["split"])
    if any(len(values) != 1 for values in group_splits.values()):
        raise ValueError("Source S-group crosses prepared splits")
    if any(len(values) != 1 for values in source_splits.values()):
        raise ValueError("Original photo hash crosses prepared splits")
    return {key: next(iter(values)) for key, values in group_splits.items()}


def box_iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, base = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, base - top)
    a = (first[2] - first[0]) * (first[3] - first[1])
    b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (a + b - intersection) if a + b - intersection else 0


def assert_crop_separation(records):
    groups, hashes, origins = defaultdict(set), defaultdict(set), defaultdict(set)
    for record in records:
        metadata = record["metadata"]
        split = metadata["split"]
        if split not in {"train", "val"}:
            raise ValueError("Test crops are refused")
        groups[metadata["source_group"]].add(split)
        hashes[metadata["pixel_sha256"]].add(split)
        origins[metadata["source_sha256"]].add(split)
    if any(len(values) != 1 for mapping in (groups, hashes, origins) for values in mapping.values()):
        raise ValueError("Cross-split group, original photo or pixel hash leakage")
    if len(hashes) != len(records):
        raise ValueError("Duplicate crop pixels")


def integral_count(integral, x, y, side):
    return int(integral[y + side, x + side] - integral[y, x + side]
               - integral[y + side, x] + integral[y, x])


def offsets(length, side):
    if length < side:
        return []
    return sorted(set([*range(0, length - side + 1, side // 2), length - side]))


def candidate_windows(mask):
    """Integral counts keep the dense audit CPU-only and inexpensive."""
    present = [int(value) for value in np.unique(mask) if 1 <= value <= 6]
    integrals = {value: cv2.integral((mask == value).astype(np.uint8), sdepth=cv2.CV_32S) for value in present}
    foreground = cv2.integral(((mask >= 1) & (mask <= 6)).astype(np.uint8), sdepth=cv2.CV_32S)
    ignored = cv2.integral((mask == 255).astype(np.uint8), sdepth=cv2.CV_32S)
    for side in (128, 256):
        for y in offsets(mask.shape[0], side):
            for x in offsets(mask.shape[1], side):
                area = side * side
                quality = {"foreground_fraction": integral_count(foreground, x, y, side) / area,
                           "ignore_fraction": integral_count(ignored, x, y, side) / area}
                if quality["foreground_fraction"] < .98 or quality["ignore_fraction"] > .005:
                    continue
                for class_id in present:
                    quality["target_area_fraction"] = integral_count(integrals[class_id], x, y, side) / area
                    if eligible_quality(quality):
                        yield class_id, (x, y, x + side, y + side), dict(quality)
                        break


def collect_candidates(prepared, rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] in {"train", "val"}:
            grouped[row["source_sha256"]].append(row)
    candidates = defaultdict(list)
    annotation_audit = []
    for index, (original_hash, tiles) in enumerate(sorted(grouped.items()), 1):
        first = tiles[0]
        annotation = source_path(prepared, first["original_annotation"])
        if digest(annotation) != first["annotation_sha256"]:
            raise ValueError("Original annotation checksum mismatch")
        with Image.open(annotation) as opened:
            original_size = opened.size
            if list(original_size) != first["original_size"]:
                raise ValueError("Annotation source dimensions differ")
            factor = first["resize_factor"]
            size = tuple(max(2, round(value * factor)) for value in original_size)
            resized = opened.convert("RGB").resize(size, Image.Resampling.NEAREST)
        exact = exact_mask(np.asarray(resized))
        annotation_audit.append({"source_sha256": original_hash, "annotation_sha256": digest(annotation),
                                 "source_group": first["well_group"], "split": first["split"],
                                 "original_publisher_split": first["source_split"],
                                 "exact_mask_ignore_fraction_after_resize": float((exact == 255).mean())})
        for row in tiles:
            if row["split"] != first["split"] or row["well_group"] != first["well_group"]:
                raise ValueError("One source image has inconsistent groups")
            tx, ty, width, height = row["tile_xywh"]
            mask = exact[ty:ty + height, tx:tx + width]
            if mask.shape != (height, width):
                raise ValueError("Tile outside source annotation")
            for class_id, box, quality in candidate_windows(mask):
                global_box = (tx + box[0], ty + box[1], tx + box[2], ty + box[3])
                candidates[row["split"], class_id].append({"row": row, "box": box,
                    "resized_source_box": global_box, "quality": quality})
        print(f"mask audit {index}/{len(grouped)}", flush=True)
    return candidates, annotation_audit


def select_candidates(candidates, seed, train_limit, val_limit):
    rng = random.Random(seed)
    selected, used_boxes = [], defaultdict(list)
    per_photo, per_group = Counter(), Counter()
    for split, limit in (("train", train_limit), ("val", val_limit)):
        for class_id, *_ in CLASSES:
            groups = defaultdict(list)
            pool = sorted(candidates[split, class_id], key=lambda c: (c["row"]["source_sha256"], c["resized_source_box"]))
            rng.shuffle(pool)
            for candidate in pool:
                groups[candidate["row"]["well_group"]].append(candidate)
            count = 0
            while groups and count < limit:
                for group in sorted(list(groups)):
                    if count >= limit:
                        break
                    accepted = None
                    while groups[group]:
                        candidate = groups[group].pop()
                        row = candidate["row"]
                        photo = row["source_sha256"]
                        if per_photo[split, class_id, photo] >= (8 if split == "train" else 4):
                            continue
                        if per_group[split, class_id, group] >= (24 if split == "train" else 6):
                            continue
                        if any(box_iou(candidate["resized_source_box"], previous) > .1 for previous in used_boxes[photo]):
                            continue
                        accepted = candidate
                        break
                    if not groups[group]:
                        del groups[group]
                    if accepted is not None:
                        selected.append((split, class_id, accepted))
                        photo = accepted["row"]["source_sha256"]
                        used_boxes[photo].append(accepted["resized_source_box"])
                        per_photo[split, class_id, photo] += 1
                        per_group[split, class_id, group] += 1
                        count += 1
    return selected


def contact_sheet(stage, records):
    # This audit sheet is separate; labels are never burned into training PNGs.
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    samples = []
    for _, code, label, _ in CLASSES:
        available = [record for record in records if record["metadata"]["source_class_code"] == code]
        chosen = [record for record in available if record["metadata"]["split"] == "train"][:2]
        chosen += [record for record in available if record["metadata"]["split"] == "val"][:2]
        if len(chosen) < 4:
            chosen += [record for record in available if record not in chosen][:4 - len(chosen)]
        samples.extend((code, label, record) for record in chosen)
    sheet = Image.new("RGB", (4 * 220, 6 * 210), "#eef3f8")
    painter = ImageDraw.Draw(sheet)
    for index, (code, label, record) in enumerate(samples):
        row, col = divmod(index, 4)
        x, y = col * 220, row * 210
        relative = record["metadata"]["relative_image"]
        with Image.open(stage / relative) as opened:
            thumb = opened.convert("RGB")
            thumb.thumbnail((204, 170))
        sheet.paste(thumb, (x + 8 + (204 - thumb.width) // 2, y + 4))
        painter.text((x + 8, y + 176), f"{code} · {record['metadata']['source_group']} · {record['metadata']['split']}", fill="#173454", font=font)
        painter.text((x + 8, y + 194), label[:26], fill="#4f657c", font=font)
    sheet.save(stage / "contact_sheet.png")


def prepare(prepared, output, seed=20261030, train_limit=64, val_limit=12):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    active_script = Path(__file__).read_bytes()
    if output.exists():
        raise ValueError("Output already exists")
    rows = read_rows(prepared / "manifest.jsonl")
    assignments = validate_source_groups(rows)
    candidates, annotation_audit = collect_candidates(prepared, rows)
    selected = select_candidates(candidates, seed, train_limit, val_limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".unibo-vlm-", dir=output.parent)).resolve()
    if stage.parent != output.parent or not stage.name.startswith(".unibo-vlm-"):
        raise ValueError("Unexpected staging directory")
    records, seen_pixels = [], {}
    classes = {value[0]: value for value in CLASSES}
    try:
        for split, class_id, candidate in selected:
            row, box = candidate["row"], candidate["box"]
            path = source_path(prepared, row["image"])
            with Image.open(path) as opened:
                if opened.size != tuple(row["tile_xywh"][2:]):
                    raise ValueError("Prepared photo/tile dimensions differ")
                crop = opened.convert("RGB").crop(box)
            pixel_hash = pixel_digest(crop)
            if pixel_hash in seen_pixels:
                if seen_pixels[pixel_hash] != split:
                    raise ValueError("Crop pixels cross splits")
                continue
            seen_pixels[pixel_hash] = split
            _, code, label, rgb = classes[class_id]
            identity = hashlib.sha256(json.dumps([row["source_sha256"], candidate["resized_source_box"], code]).encode()).hexdigest()[:24]
            relative = f"images/{split}/{code}_{identity}.png"
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)
            records.append({"image": str(output / relative),
                "conversations": [{"from": "human", "value": "<image>\n" + PROMPT},
                                  {"from": "gpt", "value": json.dumps({"facies_name": label}, ensure_ascii=False)}],
                "metadata": {"record_id": f"unibo_facies_{identity}", "record_type": "source_labeled_facies_crop",
                    "source_id": "unibo7308", "source_url": SOURCE_URL, "source_license": LICENSE,
                    "usage_scope": "noncommercial_research_only_no_automatic_deployment",
                    "label_provenance": "source_direct_mask_exact_palette", "source_class_code": code,
                    "source_class_display_ru": label, "mask_id": class_id, "mask_rgb": list(rgb),
                    "training_approved": split == "train", "approval_scope": "source_mask_research_crop_not_expert_certification",
                    "geology_approved": False, "split": split, "source_group": row["well_group"],
                    "source_group_definition": "S-prefix grouping proxy; not a verified borehole ID",
                    "source_sha256": row["source_sha256"], "source_original_split": row["source_split"],
                    "source_archive_member": row["source"], "annotation_archive_member": row["annotation"],
                    "original_annotation_sha256": row["annotation_sha256"], "relative_image": relative,
                    "image_sha256": digest(target), "pixel_sha256": pixel_hash, "image_dimensions_px": list(crop.size),
                    "prepared_tile_image": row["image"], "tile_xywh": row["tile_xywh"], "tile_crop_box": list(box),
                    "resized_source_crop_box": list(candidate["resized_source_box"]), "resize_factor": row["resize_factor"],
                    "quality": candidate["quality"], "depths_available": False}})
        assert_crop_separation(records)
        counts = {split: {code: sum(record["metadata"]["split"] == split and record["metadata"]["source_class_code"] == code for record in records)
                           for _, code, _, _ in CLASSES} for split in ("train", "val")}
        (stage / "preparation_script_snapshot.py").write_bytes(active_script)
        summary = {"source_url": SOURCE_URL, "license": LICENSE, "research_only": True, "seed": seed,
            "preparation_script_sha256": hashlib.sha256(active_script).hexdigest(),
            "counts": counts, "source_manifest_sha256": digest(prepared / "manifest.jsonl"),
            "split_strategy": "preserve_prepared_S_prefix_groups_before_cropping; S-prefix is not verified well identity",
            "source_group_assignments": assignments, "prepared_test_pixels_opened": 0,
            "original_publisher_splits_are_not_sealed": True,
            "original_publisher_split_counts_by_crop": {split: dict(Counter(record["metadata"]["source_original_split"] for record in records if record["metadata"]["split"] == split)) for split in ("train", "val")},
            "original_publisher_split_note": "Publisher folders were regrouped by S-prefix before cropping. Only the prepared S-group test is reserved and unopened; original publisher Test images may occur in new train/val.",
            "candidate_counts": {f"{split}:{classes[class_id][1]}": len(pool) for (split, class_id), pool in candidates.items()},
            "crop_sizes": [128, 256], "purity_min": .98, "foreground_min": .98, "ignore_max": .005,
            "overlap_iou_max": .1, "source_photo_class_caps": {"train": 8, "val": 4},
            "source_group_class_caps": {"train": 24, "val": 6},
            "source_groups_with_selected_crops": {split: sorted({record["metadata"]["source_group"] for record in records if record["metadata"]["split"] == split}) for split in ("train", "val")},
            "unique_source_photos": len({record["metadata"]["source_sha256"] for record in records}),
            "unique_crop_pixels": len(seen_pixels), "source_annotations_audited": annotation_audit,
            "limitations": ["Six source-domain Holocene facies; not arbitrary Russian core facies", "No depth calibration or full descriptions", "Neighbour crops are correlated", "No geologist approval is inferred"]}
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        missing = [f"{split}:{code}" for split in counts for code, count in counts[split].items() if not count]
        if missing:
            raise ValueError("Unsupported classes after strict crop selection: " + ", ".join(missing))
        for split in ("train", "val"):
            chosen = [record for record in records if record["metadata"]["split"] == split]
            (stage / f"{split}.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in chosen), encoding="utf-8")
        contact_sheet(stage, records)
        (stage / "LICENSE_SOURCES.md").write_text(
            "# Research dataset provenance\n\nAndrea Di Martino and Alessandro Amorosi (2023), Sedimentary Facies Analysis and Segmentation, University of Bologna, DOI 10.6092/unibo/amsacta/7308.\n\n"
            + SOURCE_URL + "\n\nSource licence: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/. Cropping, exact-palette conversion and Russian class translations are modifications. Preserve attribution and noncommercial/share-alike restrictions. This prepared set is a separate research experiment; do not automatically deploy into commercial CV-1 weights.\n\nThe original publisher train/validation/test folders were regrouped by S-prefix before this research experiment. Original publisher Test images can occur in the new train/val. Only the prepared source-group test is reserved and its pixels were not opened. S-prefixes are grouping proxies, not verified borehole IDs. No depths were inferred.\n",
            encoding="utf-8")
        if output.exists() or stage.parent != output.parent:
            raise ValueError("Output destination changed during preparation")
        stage.rename(output)
        return summary
    except Exception as error:
        (stage / "FAILED.txt").write_text(str(error), encoding="utf-8")
        raise RuntimeError(f"Dataset not published; quarantined staging directory: {stage}; {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "open_datasets/unibo/prepared/unibo_v1")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "vlm_core_v1/data/unibo_vlm_facies_20261030")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--val-per-class", type=int, default=12)
    args = parser.parse_args()
    summary = prepare(args.prepared, args.output, args.seed, args.train_per_class, args.val_per_class)
    print(json.dumps({"output": str(args.output.resolve()), "counts": summary["counts"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
