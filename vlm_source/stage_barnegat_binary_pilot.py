"""Stage a reviewable USGS Barnegat Bay sand/fine core-photo pilot.

The source log contains interval descriptions, and the accompanying stitched
photos have a centimetre ruler. Pixel/depth calibration here is approximate;
all generated image labels remain experimental candidates.
Run with a Python that has pypdf and Pillow installed.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "external_raw" / "usgs_barnegat_bay_2014"
OUTPUT = ROOT / "data" / "barnegat_binary_pilot"
PDF = SOURCE / "BB-SPCMSC_core-logs.pdf"
ZIP = SOURCE / "BB-SPCMSC_core-photos.zip"
INTERVAL = re.compile(r"(?m)^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*cm\s*:\s*")
CORE_ID = re.compile(r"Core ID:\s*(SPCMSC-\d+)")
CORE_LENGTH = re.compile(r"Core Length:\s*(\d+(?:\.\d+)?)\s*cm")
HOLDOUT_WELLS = {"SPCMSC-03", "SPCMSC-07", "SPCMSC-12", "SPCMSC-16", "SPCMSC-20"}
SAND = {"SAND", "MUDDY SAND", "ORGANIC-RICH SAND", "SHELLY SAND"}
FINE = {
    "MUD", "SILTY MUD", "CLAY", "ORGANIC-RICH MUD", "SILTY CLAY",
    "SILT", "SHELLY MUD", "ORGANIC-RICH CLAY", "SHELLY CLAY",
}
PROMPT = (
    "<image>\nПо этому фрагменту фотографии керна выбери преобладающий материал. "
    "Ответь ровно одним словом из списка: песчаный, тонкозернистый."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_logs() -> list[dict]:
    intervals: list[dict] = []
    for page_number, page in enumerate(PdfReader(PDF).pages, start=1):
        content = page.extract_text() or ""
        if "Lithologic Description" not in content:
            continue
        core_match = CORE_ID.search(content)
        length_match = CORE_LENGTH.search(content)
        if not core_match or not length_match:
            continue
        well = core_match.group(1)
        length = float(length_match.group(1))
        description_column = content.split("Lithologic Description", 1)[1]
        matches = list(INTERVAL.finditer(description_column))
        for number, match in enumerate(matches):
            top_cm, base_cm = map(float, match.groups())
            source_text = " ".join(
                description_column[match.end() : matches[number + 1].start() if number + 1 < len(matches) else None].split()
            )
            lithology = source_text.split(",", 1)[0].strip()
            label = "песчаный" if lithology in SAND else "тонкозернистый" if lithology in FINE else None
            intervals.append({
                "well_id": well, "page": page_number, "core_length_cm": length,
                "top_cm": top_cm, "base_cm": base_cm,
                "source_lithology": lithology, "source_description": source_text,
                "label": label,
            })
    return intervals


def extract_photos() -> dict[str, Path]:
    folder = OUTPUT / "source_photos"
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    with zipfile.ZipFile(ZIP) as archive:
        for name in archive.namelist():
            if "/SPCMSC_stitched/" not in name or not name.lower().endswith(".jpg") or name.startswith("__MACOSX"):
                continue
            filename = Path(name).name
            match = re.match(r"(SPCMSC-\d+)_Stitched_CM\.jpg$", filename, re.I)
            if not match:
                continue
            target = folder / filename
            if not target.exists():
                target.write_bytes(archive.read(name))
            paths[match.group(1)] = target
    return paths


def prepare() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    crop_dir = OUTPUT / "crops"
    crop_dir.mkdir(exist_ok=True)
    photos = extract_photos()
    intervals = parse_logs()
    rows: list[dict] = []
    exclusions = Counter()
    image_cache: dict[str, Image.Image] = {}
    try:
        for source_item in intervals:
            item = source_item
            well = item["well_id"]
            if well not in photos:
                exclusions["missing_photo"] += 1
                continue
            if item["label"] is None:
                exclusions["mixed_or_other_lithology"] += 1
                continue
            top, base, length = item["top_cm"], item["base_cm"], item["core_length_cm"]
            if base - top < 6 or not (0 <= top < base <= length + 0.5):
                exclusions["short_or_invalid_interval"] += 1
                continue
            if well not in image_cache:
                image_cache[well] = Image.open(photos[well]).convert("RGB")
            photo = image_cache[well]
            width, height = photo.size
            # Long logged intervals become short visual tiles; a 193-cm strip
            # would lose almost all texture when resized for Qwen.
            margin = min(1.5, 0.03 * (base - top)) if base - top > 20 else 0.0
            count = max(1, math.ceil((base - top - 2 * margin) / 15))
            step = (base - top - 2 * margin) / count
            for tile_number in range(count):
                tile_top = top + margin + tile_number * step
                tile_base = tile_top + step
                # Keep the centre 85% of each tile to avoid boundary mixing.
                middle = (tile_top + tile_base) / 2
                half = (tile_base - tile_top) * 0.425
                start_cm, end_cm = middle - half, middle + half
                # The stitched images have rulers from 0 cm to the logged core length.
                # One-percent edge margins are a provisional calibration, not a verified tie.
                x0 = 0.01 * width
                pixels_per_cm = 0.98 * width / length
                left = max(0, round(x0 + start_cm * pixels_per_cm))
                right = min(width, round(x0 + end_cm * pixels_per_cm))
                upper, lower = round(0.17 * height), round(0.77 * height)
                if right - left < 64 or lower - upper < 64:
                    exclusions["crop_too_small"] += 1
                    continue
                filename = f"{well}_{top:g}-{base:g}cm_tile{tile_number + 1:02d}.jpg"
                target = crop_dir / filename
                if not target.exists():
                    photo.crop((left, upper, right, lower)).save(target, quality=92)
                split = "holdout" if well in HOLDOUT_WELLS else "train"
                row = {
                "image": str(target.resolve()),
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt", "value": item["label"]},
                ],
                "metadata": {
                    "record_id": f"barnegat_{well}_{top:g}_{base:g}_tile{tile_number + 1:02d}",
                    "well_id": well,
                    "source_group": "usgs_barnegat_bay_2014",
                    "source_url": "https://pubs.usgs.gov/ds/0985/ds985_data.html",
                    "source_pdf": str(PDF.resolve()),
                    "source_pdf_page": item["page"],
                    "source_photo": str(photos[well].resolve()),
                    "source_description": item["source_description"],
                    "source_lithology": item["source_lithology"],
                    "depth_interval_cm": [top, base],
                    "tile_depth_interval_cm": [round(tile_top, 3), round(tile_base, 3)],
                    "crop_box_px": [left, upper, right, lower],
                    "split": split,
                    "training_approved": False,
                    "geology_approved": False,
                    "alignment_status": "approximate_ruler_calibration_needs_review",
                    "label_status": "source_log_binary_mapping_needs_visual_review",
                },
                }
                rows.append(row)
    finally:
        for photo in image_cache.values():
            photo.close()

    for split in ("train", "holdout"):
        selected = [r for r in rows if r["metadata"]["split"] == split]
        (OUTPUT / f"{split}_candidates.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8"
        )
    with (OUTPUT / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["split", "well_id", "top_cm", "base_cm", "label_ru", "source_lithology_en", "source_description_en", "image", "crop_box_px", "review_decision"])
        for row in rows:
            m = row["metadata"]
            writer.writerow([m["split"], m["well_id"], *m["depth_interval_cm"], row["conversations"][1]["value"], m["source_lithology"], m["source_description"], row["image"], m["crop_box_px"], ""])
    cards = []
    for row in rows:
        m = row["metadata"]
        relative = Path(row["image"]).relative_to(OUTPUT).as_posix()
        cards.append(
            "<article><img loading='lazy' src='" + html.escape(relative) + "'>"
            + f"<p><b>{html.escape(m['well_id'])} {m['depth_interval_cm'][0]:g}-{m['depth_interval_cm'][1]:g} см</b> "
            + f"[{html.escape(m['split'])}] — {html.escape(row['conversations'][1]['value'])}</p>"
            + f"<p>{html.escape(m['source_description'])}</p></article>"
        )
    (OUTPUT / "review.html").write_text(
        "<!doctype html><html lang='ru'><meta charset='utf-8'><title>Barnegat Bay — проверка пар</title>"
        "<style>body{font:16px Arial;max-width:1000px;margin:auto;padding:24px}article{border:1px solid #aaa;padding:16px;margin:16px 0}img{max-width:100%;max-height:320px}</style>"
        "<h1>Кандидаты Barnegat Bay</h1><p>Границы фото приблизительны. Сверьте каждую метку с изображением и PDF до производственного обучения.</p>"
        + "\n".join(cards) + "</html>", encoding="utf-8"
    )
    summary = {
        "source": "USGS Data Series 985 Barnegat Bay",
        "source_url": "https://pubs.usgs.gov/ds/0985/ds985_data.html",
        "photo_zip_sha256": sha256(ZIP), "log_pdf_sha256": sha256(PDF),
        "source_wells": len(photos), "source_intervals": len(intervals),
        "candidate_records": len(rows), "split_counts": dict(Counter(r["metadata"]["split"] for r in rows)),
        "class_counts": dict(Counter(r["conversations"][1]["value"] for r in rows)),
        "exclusions": dict(exclusions), "holdout_wells": sorted(HOLDOUT_WELLS),
        "status": "candidate_only_not_expert_approved_not_integrated",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
