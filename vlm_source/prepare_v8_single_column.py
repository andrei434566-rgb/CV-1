"""Rebuild R-41 depth crops using only the original one-metre, single-column photos."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image

from prepare_interval_crop_training import first_sentence, make_prompt


ROOT = Path(__file__).resolve().parent
PAIRS = ROOT / "data" / "candidate_pairs.jsonl"
TEXT = ROOT / "data" / "knowledge" / "text_knowledge_training_v1.jsonl"
OLD_TEST = ROOT / "data" / "v7_heldout" / "test_visual.jsonl"
OUT = ROOT / "data" / "v8_single_column"
IMAGES = OUT / "images"
MAX_TILE_METERS = 0.18
MAX_TILES = 5
MIN_NATIVE_HEIGHT = 900
PAD_METERS = 0.005


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    pairs = read(PAIRS)
    best = {}
    excluded_multi = 0
    for pair in pairs:
        source_image = Path(pair["image"]["local_copy"])
        with Image.open(source_image) as opened:
            width, height = opened.size
        if height < 10000:
            excluded_multi += 1
            continue
        photo_top = float(pair["image_depth_interval_m"]["top"])
        photo_base = float(pair["image_depth_interval_m"]["base"])
        for segment in pair["source_description_segments"]:
            top = max(photo_top, float(segment["top_m"]))
            base = min(photo_base, float(segment["base_m"]))
            if base <= top:
                continue
            key = int(segment["source_row"])
            overlap = base - top
            if key not in best or overlap > best[key][0]:
                best[key] = (overlap, pair, segment, top, base)
    if len(best) != 74:
        raise ValueError(f"Expected all 74 description rows in single-column photos, found {len(best)}")

    visual = []
    skipped_short = 0
    source_photos = set()
    for row_number, (_, pair, segment, top, base) in sorted(best.items()):
        path = Path(pair["image"]["local_copy"])
        photo_top = float(pair["image_depth_interval_m"]["top"])
        photo_base = float(pair["image_depth_interval_m"]["base"])
        source_photos.add(pair["id"])
        description = first_sentence(str(segment["source_description"]))
        count = min(MAX_TILES, max(1, math.ceil((base - top) / MAX_TILE_METERS)))
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        for tile_index in range(count):
            tile_top = top + (base - top) * tile_index / count
            tile_base = top + (base - top) * (tile_index + 1) / count
            crop_top = max(photo_top, tile_top - PAD_METERS)
            crop_base = min(photo_base, tile_base + PAD_METERS)
            y0 = max(0, min(height - 2, round((crop_top - photo_top) / (photo_base - photo_top) * height)))
            y1 = max(y0 + 2, min(height, round((crop_base - photo_top) / (photo_base - photo_top) * height)))
            if y1 - y0 < MIN_NATIVE_HEIGHT:
                skipped_short += 1
                continue
            # Single-column images contain one continuous core track; small left/right margins are trimmed.
            x0, x1 = round(width * 0.07), round(width * 0.93)
            crop = image.crop((x0, y0, x1, y1))
            image_out = IMAGES / f"r41_row{row_number:03d}_tile{tile_index + 1}.jpg"
            crop.save(image_out, format="JPEG", quality=95, optimize=True)
            target = {"description_segments": [{
                "top_m": round(tile_top, 3), "base_m": round(tile_base, 3),
                "description": description,
            }]}
            visual.append({
                "image": str(image_out),
                "conversations": [
                    {"from": "human", "value": make_prompt(tile_top, tile_base, top, base)},
                    {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
                ],
                "metadata": {
                    "record_id": f"r41_v8_row{row_number}_tile{tile_index + 1}",
                    "record_type": "user_confirmed_single_column_crop",
                    "training_approved": False,
                    "source_pair_approved_by_user": True,
                    "geology_approved": False,
                    "crop_reviewed_by_geologist": False,
                    "depth_alignment_method": "linear_interpolation_from_photo_endpoints",
                    "well_id": "R-41", "source_well": "R-41",
                    "source_group": "E_drive_real_core_R41",
                    "source_row": row_number,
                    "source_photo_record_id": pair["id"],
                    "source_photo": str(path),
                    "source_workbook": pair["provenance"]["description_workbook"],
                    "source_description_full": segment["source_description"],
                    "source_description_interval_m": [top, base],
                    "photo_tile_depth_interval_m": [round(tile_top, 3), round(tile_base, 3)],
                    "source_photo_interval_m": [photo_top, photo_base],
                    "crop_native_size_px": list(crop.size),
                    "single_column_verified_by_source_dimensions": True,
                    "target_uses_first_sentence_only": True,
                },
            })
    text_records = [r for r in read(TEXT) if r.get("metadata", {}).get("training_approved") is True]
    old_test_rows = {r["metadata"]["source_row"] for r in read(OLD_TEST)}
    test_photos = {r["metadata"]["source_photo_record_id"] for r in visual if r["metadata"]["source_row"] in old_test_rows}
    train = [r for r in visual if r["metadata"]["source_photo_record_id"] not in test_photos]
    test = [r for r in visual if r["metadata"]["source_photo_record_id"] in test_photos]
    shared = [r for r in test if r["metadata"]["source_row"] in old_test_rows]
    if {r["metadata"]["source_row"] for r in train} & {r["metadata"]["source_row"] for r in test}:
        raise ValueError("Description row crossed the corrected photo split")
    write(OUT / "all_visual.jsonl", visual)
    write(OUT / "train_visual.jsonl", train)
    write(OUT / "train_mixed.jsonl", train + text_records)
    write(OUT / "test_visual.jsonl", test)
    write(OUT / "test_shared_v7_rows.jsonl", shared)
    summary = {
        "source": str(PAIRS), "source_description_rows": len(best),
        "excluded_multi_column_source_photos": excluded_multi,
        "single_column_source_photos_selected": len(source_photos),
        "visual_crops": len(visual), "skipped_too_short_tiles": skipped_short,
        "train_crops": len(train), "train_rows": len({r["metadata"]["source_row"] for r in train}),
        "test_crops": len(test), "test_rows": len({r["metadata"]["source_row"] for r in test}),
        "shared_v7_test_rows": len({r["metadata"]["source_row"] for r in shared}),
        "text_only_train_rows": len(text_records),
        "train_test_photo_overlap": len({r["metadata"]["source_photo_record_id"] for r in train} & test_photos),
        "native_crop_aspect_counts": dict(Counter(
            "very_wide" if r["metadata"]["crop_native_size_px"][0] / r["metadata"]["crop_native_size_px"][1] > 3 else "other"
            for r in visual
        )),
        "limitation": "These experimental crops are not training-approved at tile level. Linear depth alignment and repeated full-layer first-sentence labels require geological review; train and test remain from one well.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
