"""Stage, but do not approve, the R-390 photo and old Word report candidates."""

from __future__ import annotations

import io
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
SOURCE = Path("E:/Керн фото/2/390/2. Литология")
REPORT = SOURCE / "1.2 Поинтервальное описание керна скв.390 Усть-Ямсовейское..doc"
PHOTOS = SOURCE / "1.3 Фото"
OUTPUT = ROOT / "data" / "r390_review_candidates.jsonl"
SUMMARY = ROOT / "data" / "r390_review_summary.json"
TESSERACT = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
INTERVAL = re.compile(r"(\d{4})[,.](\d+)[-–](\d{4})[,.](\d+)")


def normalize(match: re.Match) -> tuple[float, float]:
    return float(match.group(1) + "." + match.group(2)), float(match.group(3) + "." + match.group(4))


def description_sections() -> dict[tuple[float, float], dict]:
    # This particular legacy .doc stores the Russian main text as uncompressed UTF-16LE.
    # A scan of readable runs preserves its sequence without attempting a general DOC parser.
    decoded = REPORT.read_bytes().decode("utf-16le", errors="ignore")
    sections = {}
    headers = list(re.finditer(r"Интервал\s+" + INTERVAL.pattern, decoded))
    for i, match in enumerate(headers):
        depths = normalize(INTERVAL.search(match.group()))
        end = headers[i + 1].start() if i + 1 < len(headers) else len(decoded)
        runs = re.findall(r"[А-Яа-яЁё0-9.,;:!?\-–() /\n\r]{20,}", decoded[match.end():end])
        readable = [run.strip() for run in runs if len(re.findall(r"[А-Яа-яЁё]", run)) >= 15]
        body = "\n".join(readable)
        sections[depths] = {
            "header": match.group(),
            "text": body[:14000],
            "text_characters": len(body),
            "truncated": len(body) > 14000,
        }
    return sections


def read_photo_label(path: Path) -> tuple[str, tuple[float, float] | None]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    # Depth interval is printed vertically in the white margin on these figure pages.
    regions = [
        image.crop((int(width * .05), int(height * .10), int(width * .33), int(height * .90))),
        image.crop((int(width * .68), int(height * .08), int(width * .95), int(height * .90))),
    ]
    parts = []
    for region in regions:
        stream = io.BytesIO()
        region.rotate(-90, expand=True).save(stream, format="PNG")
        result = subprocess.run(
            [str(TESSERACT), "stdin", "stdout", "-l", "eng", "--psm", "11"],
            input=stream.getvalue(), capture_output=True, check=False,
        )
        text = result.stdout.decode("utf-8", errors="replace")
        parts.append(text)
        match = INTERVAL.search(text)
        if match:
            return "\n".join(parts)[:1500], normalize(match)
    return "\n".join(parts)[:1500], None


def main() -> None:
    if not TESSERACT.exists():
        raise FileNotFoundError(TESSERACT)
    sections = description_sections()
    rows = []
    photos = sorted(PHOTOS.glob("Рис *.jpg"), key=lambda path: int(re.search(r"\d+", path.stem).group()))
    for photo in photos:
        ocr, depth = read_photo_label(photo)
        section = sections.get(depth) if depth else None
        rows.append({
            "record_id": "r390_" + re.search(r"\d+", photo.stem).group().zfill(3),
            "well_id": "390", "image": str(photo),
            "image_depth_interval_m": depth,
            "image_depth_source": "printed_photo_label_tesseract_ocr" if depth else "not_read",
            "ocr_excerpt": ocr,
            "description_source": str(REPORT),
            "description_interval_m": depth if section else None,
            "source_description": section["text"] if section else None,
            "source_description_characters": section["text_characters"] if section else 0,
            "pairing_status": "candidate_whole_run_not_layer_review_required" if section else "unmatched",
            "training_approved": False,
            "reason": "The report describes a whole coring run while each figure shows a subset or an alternate view. Layer-level alignment and text extraction need expert review.",
        })
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "well": "390", "photos": len(rows), "report_intervals": len(sections),
        "report_depth_intervals_m": [list(depth) for depth in sections],
        "photo_labels_read": sum(row["image_depth_interval_m"] is not None for row in rows),
        "candidate_run_links": sum(row["pairing_status"].startswith("candidate") for row in rows),
        "training_approved": 0, "manifest": str(OUTPUT),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
