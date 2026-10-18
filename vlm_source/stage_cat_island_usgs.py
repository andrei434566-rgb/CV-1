"""Stage depth-linked Cat Island core photos and source lithologic descriptions.

These are research candidates.  Automatic PDF extraction and column cropping
must be checked against the original logs before any candidate is approved as
a visual training target.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re

import pdfplumber
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "external_raw" / "usgs_2010_cat_island"
OUT = ROOT / "data" / "cat_island_candidates"
PHOTO_DIR = OUT / "images"
MANIFEST = OUT / "source_candidates.jsonl"
SUMMARY = OUT / "summary.json"
DOI = "10.3133/ds834"
URL = "https://pubs.usgs.gov/ds/0834/ds834_data.html"
LAYER_RE = re.compile(r"(?m)(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*m\s*:")
LENGTH_RE = re.compile(r"CORE\s+LENGTH\s*\(m\)\s*:\s*(\d+(?:\.\d+)?)", re.I)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(pdf: Path) -> tuple[float | None, list[dict], str]:
    with pdfplumber.open(pdf) as document:
        page = document.pages[-1]
        full = page.extract_text() or ""
        match = LENGTH_RE.search(full)
        length = float(match.group(1)) if match else None
        # Descriptions are in the right-hand column. This crop removes the
        # lithologic/axis graphics that otherwise interleave with the text.
        right = page.crop((320, 170, page.width, min(680, page.height))).extract_text() or ""
    heading = right.find("Lithologic Description")
    description = right[heading + len("Lithologic Description"):] if heading >= 0 else ""
    starts = list(LAYER_RE.finditer(description))
    layers = []
    for index, item in enumerate(starts):
        top, base = float(item.group(1)), float(item.group(2))
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(description)
        text = " ".join(description[item.end():stop].split()).split("* Sand content", 1)[0].strip()
        if top < base and text:
            layers.append({"top_m": top, "base_m": base, "description": text})
    if length is None and layers:
        # Two marine PDF headers omit a machine-readable length.  Their final
        # lithologic interval gives the recorded depth, with provenance kept
        # separately in the summary below.
        length = layers[-1]["base_m"]
    return length, layers, right


def image_columns(pdf: Path, length: float | None) -> list[tuple[Image.Image, float, float]]:
    reader = PdfReader(pdf)
    images = reader.pages[0].images
    if not images:
        return []
    marine = "10BIM06" in pdf.name
    result = []
    if marine:
        with Image.open(io.BytesIO(images[0].data)) as opened:
            original = opened.convert("RGB")
        columns = round(original.width / 170)
        if not 3 <= columns <= 7 or length is None:
            return []
        for index in range(columns):
            left = round(index * original.width / columns)
            right = round((index + 1) * original.width / columns)
            top, base = float(index), min(float(index + 1), length)
            if base > top:
                result.append((original.crop((left, 0, right, original.height)), top, base))
    else:
        for index, embedded in enumerate(images):
            with Image.open(io.BytesIO(embedded.data)) as opened:
                image = opened.convert("RGB")
            top, base = float(index), min(float(index + 1), length or float(index + 1))
            if base > top:
                result.append((image, top, base))
    return result


def main() -> None:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted((SOURCE / "extracted").rglob("*_core_log.pdf"))
    records = []
    summaries = []
    for pdf in pdfs:
        length, layers, raw_text = describe(pdf)
        columns = image_columns(pdf, length)
        key = pdf.stem.removesuffix("_core_log")
        setting = "marine" if "10BIM06" in key else "terrestrial_beach_ridge"
        gaps = [(a["base_m"], b["top_m"]) for a, b in zip(layers, layers[1:])
                if b["top_m"] - a["base_m"] > 0.05]
        if layers and layers[0]["top_m"] > 0.05:
            gaps.insert(0, (0.0, layers[0]["top_m"]))
        summaries.append({"core_id": key, "setting": setting, "length_m": length,
                          "description_layers": len(layers), "photo_columns": len(columns),
                          "description_gaps_m": gaps,
                          "source_pdf": str(pdf), "source_pdf_sha256": digest(pdf),
                          "parsed_layers": layers, "raw_description_column": raw_text})
        for index, (image, top, base) in enumerate(columns):
            filename = PHOTO_DIR / f"{key}_{index:02d}_{top:g}-{base:g}m.jpg"
            image.save(filename, quality=95)
            clipped = [
                {"top_m": round(max(layer["top_m"], top), 3),
                 "base_m": round(min(layer["base_m"], base), 3),
                 "description": layer["description"]}
                for layer in layers
                if min(layer["base_m"], base) - max(layer["top_m"], top) > 0.02
            ]
            record = {
                "image": str(filename),
                "conversations": [
                    {"from": "human", "value": f"<image>\nОпиши керн на интервале {top:g}–{base:g} м по фото. Верни JSON с description_segments."},
                    {"from": "gpt", "value": json.dumps({"description_segments": clipped}, ensure_ascii=False)},
                ],
                "metadata": {
                    "record_id": f"cat_island:{key}:{index}", "dataset": "usgs_2010_cat_island",
                    "source_well": key, "setting_context": setting, "source_pdf": str(pdf),
                    "source_pdf_page_photo": 1, "source_pdf_page_description": 2,
                    "source_url": URL, "source_doi": DOI, "photo_interval_m": [top, base],
                    "source_language": "en", "target_language": "en",
                    "training_approved": False, "review_status": "automatic_depth_crop_and_text_parse_requires_geologist_review",
                    "source_log_has_depth_gaps": bool(gaps),
                    "image_sha256": digest(filename),
                },
            }
            records.append(record)
    MANIFEST.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    SUMMARY.write_text(json.dumps({"source_url": URL, "source_doi": DOI,
                                  "cores": summaries, "photo_candidates": len(records),
                                  "nonempty_candidates": sum(bool(json.loads(row["conversations"][1]["value"])["description_segments"]) for row in records),
                                  "training_approved": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cores": len(summaries), "photo_candidates": len(records),
                      "nonempty_candidates": sum(bool(json.loads(row["conversations"][1]["value"])["description_segments"]) for row in records),
                      "empty_or_unparsed_cores": [s["core_id"] for s in summaries if not s["parsed_layers"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
