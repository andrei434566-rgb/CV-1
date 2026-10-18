"""Crop source-labeled Midnight Pass intervals, quarantined for visual QA.

The lithology and boundaries below are manually transcribed from the seven
official log JPGs. Ruler calibration is estimated from repeated 1-cm ticks;
the saved crop omits 1.5 cm at every interval edge. Nothing is approved for
training by this script.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import statistics
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data/external_raw/usgs_midnight_pass"
OUTPUT = PROJECT / "data/usgs_midnight_pass_interval_candidates"
PHOTOS = SOURCE / "2024-330-FA_MP_CorePhotos.zip"
SOURCE_URL = "https://www.usgs.gov/data/sediment-data-cores-collected-2025-midnight-pass-florida"

# Source log labels only, not claims that all attributes are visible in a crop.
INTERVALS: dict[str, list[tuple[float, float, str, str]]] = {
    "MP2-D": [(0, 10.5, "SAND", "песок"),
              (10.5, 67.5, "SAND and SHELLY SAND", "песок и песок с раковинным материалом")],
    "MP2-E": [(0, 30, "SHELLY SAND", "песок с раковинным материалом"),
              (30, 53.5, "SHELLY SAND", "песок с раковинным материалом"),
              (53.5, 94, "SAND and SHELLY SAND", "песок и песок с раковинным материалом")],
    "MP2-F": [(0, 73, "SHELL LAG", "слой раковинного материала")],
    "MP2-G": [(0, 17, "SHELL LAG", "слой раковинного материала"),
              (17, 53, "SHELL LAG", "слой раковинного материала"),
              (53, 72, "SHELL LAG", "слой раковинного материала")],
    "MP3-B": [(0, 11, "SHELL LAG", "слой раковинного материала"),
              (11, 21, "SHELLY SAND", "песок с раковинным материалом"),
              (21, 29, "SHELL LAG", "слой раковинного материала"),
              (29, 41, "SHELL LAG", "слой раковинного материала"),
              (41, 54, "SHELL LAG", "слой раковинного материала")],
    "MP3-D": [(0, 52, "SHELL LAG", "слой раковинного материала"),
              (52, 74.5, "SHELLY SAND", "песок с раковинным материалом")],
    "MP3-E": [(0, 23, "SHELL LAG", "слой раковинного материала"),
              (23, 35, "SHELLY SAND", "песок с раковинным материалом"),
              (35, 94, "SAND and SHELLY SAND", "песок и песок с раковинным материалом"),
              (94, 105.5, "SAND", "песок")],
}


def ruler_calibration(preview: Image.Image, core_length_cm: float) -> dict:
    gray = np.asarray(preview.convert("L"))
    height, width = gray.shape
    dark_fraction = (gray[int(height * .81):int(height * .87), :] < 90).mean(axis=0)
    mask = dark_fraction > .45
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    peaks = [int((start + end) / 2) for start, end in zip(starts, ends) if end - start < 10]
    approximate = width / core_length_cm
    gaps = [b - a for a, b in zip(peaks, peaks[1:])
            if approximate * .65 <= b - a <= approximate * 1.35]
    if len(gaps) < 12:
        raise ValueError(f"Too few ruler intervals ({len(gaps)})")
    step = statistics.median(gaps)
    first = next((peak for peak in peaks if peak > 1.5 * step), None)
    if first is None:
        raise ValueError("Could not locate first ruler tick")
    x_zero = first - step
    x_base = x_zero + core_length_cm * step
    if not (-.01 * width <= x_zero <= .05 * width and .94 * width <= x_base <= 1.04 * width):
        raise ValueError(f"Implausible ruler fit: x0={x_zero}, xbase={x_base}, width={width}")
    return {"preview_width_px": width, "preview_height_px": height,
            "tick_gap_count": len(gaps), "cm_per_tick": 1,
            "pixels_per_cm_preview": step, "x_at_zero_cm_preview": x_zero,
            "x_at_core_base_preview": x_base, "endpoint_fraction_of_width": x_base / width,
            "method": "median distance between visible 1-cm ruler ticks; first tick assumed 1 cm",
            "status": "candidate_requires_visual_confirmation"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    crop_dir = OUTPUT / "crops"
    crop_dir.mkdir(exist_ok=True)
    records = []
    calibrations = {}
    with zipfile.ZipFile(PHOTOS) as archive:
        for core_id, intervals in INTERVALS.items():
            length_cm = intervals[-1][1]
            member = f"{core_id}.jpg"
            with Image.open(io.BytesIO(archive.read(member))) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            preview_file = SOURCE / "review_previews" / f"photo_{core_id}.jpg"
            with Image.open(preview_file) as preview:
                fit = ruler_calibration(preview, length_cm)
            ratio = image.width / fit["preview_width_px"]
            calibrations[core_id] = {"source_photo_member": member,
                                     "source_log_member": f"2024-330-FA-{core_id}.jpg",
                                     "core_length_cm": length_cm,
                                     "full_resolution_px": list(image.size), **fit}
            for number, (top, base, source_label, russian_candidate) in enumerate(intervals, 1):
                inner_top, inner_base = top + 1.5, base - 1.5
                if inner_base <= inner_top:
                    raise ValueError(f"Interval too short: {core_id} {top}-{base}")
                left = round((fit["x_at_zero_cm_preview"] + inner_top * fit["pixels_per_cm_preview"]) * ratio)
                right = round((fit["x_at_zero_cm_preview"] + inner_base * fit["pixels_per_cm_preview"]) * ratio)
                upper, lower = round(image.height * .025), round(image.height * .69)
                if not (0 <= left < right <= image.width and 0 <= upper < lower <= image.height):
                    raise ValueError(f"Invalid crop bounds: {core_id} {top}-{base}")
                filename = f"{core_id}_{number:02d}_{top:g}-{base:g}cm.jpg"
                destination = crop_dir / filename
                if not destination.exists():
                    image.crop((left, upper, right, lower)).save(destination, quality=92, optimize=True)
                records.append({"record_id": f"usgs_midnight_{core_id}_{number:02d}",
                                "source_url": SOURCE_URL, "source_license": "CC0 1.0 Universal",
                                "core_id": core_id, "source_photo_member": member,
                                "source_log_member": f"2024-330-FA-{core_id}.jpg",
                                "source_interval_cm": [top, base],
                                "source_lithology_en": source_label,
                                "russian_target_candidate": russian_candidate,
                                "crop_interior_cm": [inner_top, inner_base],
                                "crop_box_full_resolution_px": [left, upper, right, lower],
                                "crop_path": str(destination.resolve()),
                                "crop_sha256": sha256(destination),
                                "ruler_calibration_status": fit["status"],
                                "label_scope": "dominant source lithology only; full source description is not a photo target",
                                "training_approved": False})
    report = {"source_url": SOURCE_URL, "license": "CC0 1.0 Universal",
              "source_photo_archive_sha256": sha256(PHOTOS),
              "manual_log_lithology_transcription": True,
              "ruler_calibration_status": "candidate_requires_visual_confirmation",
              "cores": calibrations, "interval_candidates": records,
              "training_approved_count": 0}
    (OUTPUT / "candidate_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for record in records:
        name = Path(record["crop_path"]).name
        top, base = record["source_interval_cm"]
        cards.append(f"<figure><img loading='lazy' src='crops/{html.escape(name)}'>"
                     f"<figcaption><b>{html.escape(record['core_id'])} {top:g}–{base:g} см</b><br>"
                     f"Источник: {html.escape(record['source_lithology_en'])}<br>"
                     f"Кандидат: {html.escape(record['russian_target_candidate'])}</figcaption></figure>")
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>Midnight Pass — интервалы</title>
<style>body{font:15px system-ui;max-width:1500px;margin:2rem auto;color:#20313d;background:#f7f9fb}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem}figure{margin:0;padding:.75rem;background:white;border:1px solid #ccd;border-radius:.5rem}img{width:100%;height:210px;object-fit:contain;background:#222}figcaption{line-height:1.5}</style>
<h1>USGS Midnight Pass — 20 кандидатов</h1><p>Литология и границы перенесены из официальных журналов. Координаты по линейке вычислены автоматически; у каждого контакта исключено 1,5 см. Проверьте калибровку, видимость признаков и русский текст до любого обучения. Утверждённых пар: 0.</p><main>""" + "".join(cards) + "</main></html>"
    (OUTPUT / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"cores": len(calibrations), "interval_candidates": len(records),
                      "training_approved": 0,
                      "ruler_endpoint_fraction": {key: round(value["endpoint_fraction_of_width"], 3)
                                                  for key, value in calibrations.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
