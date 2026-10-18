"""Create interval-crop image/text pairs from confirmed measured-depth records."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
PAIR_INPUT = DATA / "candidate_pairs.jsonl"
OUTPUT_DIR = DATA / "e_drive_interval_crops_v2"
IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST = OUTPUT_DIR / "qwen3_vl_interval_crops.jsonl"
SUMMARY = OUTPUT_DIR / "summary.json"
TEXT_KNOWLEDGE_SOURCE = DATA / "qwen3_v4_e_drive_training.jsonl"

MAX_TILE_METERS = 0.25
MAX_TILES_PER_LAYER = 3
DEPTH_PADDING_METERS = 0.005
EDGE_PADDING_FRACTION = 0.02
JPEG_QUALITY = 96


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_sentence(description: str) -> str:
    text = " ".join(description.split()).strip()
    pieces = re.split(r"\.\s+(?=[А-ЯЁ])", text, maxsplit=1)
    return pieces[0].rstrip(".; ") + ("." if pieces else "")


def intervals_for_segment(top: float, base: float) -> list[tuple[float, float]]:
    length = base - top
    count = min(MAX_TILES_PER_LAYER, max(1, math.ceil(length / MAX_TILE_METERS)))
    step = length / count
    return [(top + i * step, base if i == count - 1 else top + (i + 1) * step) for i in range(count)]


def make_prompt(tile_top: float, tile_base: float, layer_top: float, layer_base: float) -> str:
    return (
        "<image>\n"
        f"Это увеличенный фрагмент керна внутри описываемого слоя {layer_top:.2f}–{layer_base:.2f} м; "
        f"на фото показан участок примерно {tile_top:.2f}–{tile_base:.2f} м. "
        "Составь краткое нейтральное литологическое описание на русском языке по непосредственно "
        "видимым признакам. Верни только JSON с ключом description_segments и одним объектом "
        "с top_m, base_m и description. Не назначай среду осадконакопления и не придумывай "
        "свойства, которых нет в видимой части. Игнорируй номера образцов и надписи на фото."
    )


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    unique_source_rows: set[tuple[str, int]] = set()
    source_photos: set[str] = set()
    dimensions: list[tuple[int, int]] = []
    counter = 0
    best_by_source_row: dict[tuple[str, int], tuple[float, dict, dict, float, float]] = {}

    for pair in read_jsonl(PAIR_INPUT):
        photo_top = float(pair["image_depth_interval_m"]["top"])
        photo_base = float(pair["image_depth_interval_m"]["base"])
        for segment_index, segment in enumerate(pair["source_description_segments"]):
            layer_top = max(photo_top, float(segment["top_m"]))
            layer_base = min(photo_base, float(segment["base_m"]))
            if layer_base <= layer_top:
                continue
            key = (pair["provenance"]["description_workbook"], int(segment["source_row"]))
            span = layer_base - layer_top
            previous = best_by_source_row.get(key)
            if previous is None or span > previous[0]:
                best_by_source_row[key] = (span, pair, segment, layer_top, layer_base)

    for _, pair, segment, layer_top, layer_base in best_by_source_row.values():
        image_path = Path(pair["image"]["local_copy"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing source photo: {image_path}")
        photo_top = float(pair["image_depth_interval_m"]["top"])
        photo_base = float(pair["image_depth_interval_m"]["base"])
        if photo_base <= photo_top:
            continue
        source_photos.add(pair["id"])

        with Image.open(image_path) as opened:
            original = opened.convert("RGB")
        width, height = original.size
        dimensions.append((width, height))
        x_pad = round(width * EDGE_PADDING_FRACTION)

        source_row = int(segment["source_row"])
        source_workbook = pair["provenance"]["description_workbook"]
        unique_source_rows.add((source_workbook, source_row))
        description = first_sentence(str(segment["source_description"]))
        if len(description) < 8:
            continue

        for tile_index, (tile_top, tile_base) in enumerate(intervals_for_segment(layer_top, layer_base), start=1):
            crop_top_depth = max(photo_top, tile_top - DEPTH_PADDING_METERS)
            crop_base_depth = min(photo_base, tile_base + DEPTH_PADDING_METERS)
            y0 = max(0, min(height - 2, round((crop_top_depth - photo_top) / (photo_base - photo_top) * height)))
            y1 = max(y0 + 2, min(height, round((crop_base_depth - photo_top) / (photo_base - photo_top) * height)))
            if y1 - y0 < 32:
                continue
            crop = original.crop((x_pad, y0, width - x_pad, y1))
            counter += 1
            filename = f"crop_{counter:05d}.jpg"
            output_image = IMAGE_DIR / filename
            crop.save(output_image, format="JPEG", quality=JPEG_QUALITY, optimize=True)

            target = {
                "description_segments": [
                    {
                        "top_m": round(tile_top, 3),
                        "base_m": round(tile_base, 3),
                        "description": description,
                    }
                ]
            }
            records.append(
                {
                    "image": str(output_image),
                    "conversations": [
                        {"from": "human", "value": make_prompt(tile_top, tile_base, layer_top, layer_base)},
                        {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
                    ],
                    "metadata": {
                        "record_id": f"{pair['id']}_row{source_row}_tile{tile_index}",
                        "record_type": "user_confirmed_interval_crop",
                        "training_approved": True,
                        "geology_approved": True,
                        "target_language": "ru",
                        "source_group": "E_drive_real_core_R41",
                        "source_well": "R-41",
                        "well_id": "R-41",
                        "source_photo_record_id": pair["id"],
                        "source_photo": str(image_path),
                        "source_image_depth_interval_m": [photo_top, photo_base],
                        "source_description_interval_m": [layer_top, layer_base],
                        "photo_tile_depth_interval_m": [round(tile_top, 3), round(tile_base, 3)],
                        "source_row": source_row,
                        "source_workbook": source_workbook,
                        "source_description_full": segment["source_description"],
                        "target_uses_first_sentence_only": True,
                        "pairing_approved_by_user": True,
                        "approval_id": "user_batch_confirmation_2026-10-07",
                    },
                }
            )

    visual_record_count = len(records)
    text_records = [
        row
        for row in read_jsonl(TEXT_KNOWLEDGE_SOURCE)
        if row.get("metadata", {}).get("record_type") == "text_only_knowledge"
        and row.get("metadata", {}).get("training_approved") is True
    ]
    records.extend(text_records)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "visual_records": visual_record_count,
        "text_only_knowledge_records": len(text_records),
        "total_training_records": len(records),
        "unique_source_interval_rows_selected": len(best_by_source_row),
        "source_photos": len(source_photos),
        "unique_source_description_rows": len(unique_source_rows),
        "max_tile_depth_m": MAX_TILE_METERS,
        "depth_padding_m": DEPTH_PADDING_METERS,
        "horizontal_edge_padding_fraction": EDGE_PADDING_FRACTION,
        "source_image_width_range_px": [min(w for w, _ in dimensions), max(w for w, _ in dimensions)],
        "source_image_height_range_px": [min(h for _, h in dimensions), max(h for _, h in dimensions)],
        "manifest": str(MANIFEST.relative_to(PROJECT)),
        "images": str(IMAGE_DIR.relative_to(PROJECT)),
        "note": (
            "One best-overlap photo was selected per source interval row, then each interval was split into at most three vertical tiles to preserve texture at low VRAM. "
            "Visual targets contain only the first source sentence; full original descriptions are retained in metadata. "
            "Approved Russian text-only geology examples from the v4 manifest are included to retain terminology knowledge."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
