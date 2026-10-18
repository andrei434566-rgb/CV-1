"""Inventory the official USGS DS850 core photo/log/X-ray archives.

Run with the bundled workspace Python (pypdf and Pillow required). The output
is a source-linked review set, not an automatically approved training set.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "external_raw" / "usgs_chandeleur_2012"
REVIEW = SOURCE / "review"
ARCHIVES = {
    "logs": SOURCE / "12BIM01_Core-Logs.zip",
    "photos": SOURCE / "12BIM01_Core-Photos.zip",
    "xrays": SOURCE / "12BIM01_Core-X-Rays.zip",
}
ID_RE = re.compile(r"12BIM01-\d{2}")
INTERVAL_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)?)\s*(?:cm\s*)?[-–]\s*(\d+(?:\.\d+)?)\s*cm\s*[:;\-]\s*")
OFFICIAL = "https://pubs.usgs.gov/ds/0850/html/ds850_data.html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def names_by_id(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    result = {}
    for name in archive.namelist():
        if not name.lower().endswith(suffix):
            continue
        core_id = ID_RE.search(PurePosixPath(name).name)
        if core_id is None:
            continue
        if core_id.group() in result:
            raise ValueError(f"Duplicate {suffix} for {core_id.group()}")
        result[core_id.group()] = name
    return result


def description_from_pdf(raw_pdf: bytes) -> tuple[int, str, list[dict], float | None]:
    reader = PdfReader(io.BytesIO(raw_pdf))
    page_texts = [(page.extract_text() or "") for page in reader.pages]
    page = next((text for text in page_texts if "Lithologic Description" in text), "")
    if not page:
        return len(reader.pages), "", [], None
    # The descriptive text precedes the plotted depth ticks on these source
    # sheets. A literal 0/5 tick sequence is distinct from interval headers.
    narrative = re.split(r"\n0\s*\n5\s*\n", page, maxsplit=1)[0]
    narrative = narrative.split("Lithologic Description", 1)[0].strip()
    matches = list(INTERVAL_RE.finditer(narrative))
    intervals = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(narrative)
        intervals.append({
            "top_cm": float(match.group(1)),
            "base_cm": float(match.group(2)),
            "description_en": re.sub(r"\s+", " ", narrative[match.end():end]).strip(),
        })
    length_match = (re.search(r"([0-9]+(?:\.[0-9]+)?)\s*cm\s*Core Length", page)
                    or re.search(r"Core Length:\s*([0-9]+(?:\.[0-9]+)?)\s*cm", page)
                    or re.search(r"([0-9]+(?:\.[0-9]+)?)\s*Core Length", page))
    core_length = float(length_match.group(1)) if length_match else None
    return len(reader.pages), re.sub(r"\s+", " ", narrative), intervals, core_length


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for path in ARCHIVES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "photos").mkdir(exist_ok=True)
    (REVIEW / "logs").mkdir(exist_ok=True)

    with zipfile.ZipFile(ARCHIVES["logs"]) as logs, \
         zipfile.ZipFile(ARCHIVES["photos"]) as photos, \
         zipfile.ZipFile(ARCHIVES["xrays"]) as xrays:
        crc = {
            "logs": logs.testzip(),
            "photos": photos.testzip(),
            "xrays": xrays.testzip(),
        }
        if any(crc.values()):
            raise ValueError(f"ZIP CRC failed: {crc}")
        log_names = names_by_id(logs, ".pdf")
        log_names.pop("12BIM01_all", None)
        photo_names = names_by_id(photos, ".jpg")
        xray_names = defaultdict(list)
        for name in xrays.namelist():
            match = ID_RE.search(PurePosixPath(name).name)
            if match and name.lower().endswith((".tif", ".tiff")):
                xray_names[match.group()].append(name)
        if set(log_names) != set(photo_names):
            raise ValueError(f"Log/photo core IDs differ: logs-only={set(log_names)-set(photo_names)}, photos-only={set(photo_names)-set(log_names)}")

        records = []
        for core_id in sorted(log_names):
            log_bytes = logs.read(log_names[core_id])
            photo_bytes = photos.read(photo_names[core_id])
            pages, narrative, intervals, core_length = description_from_pdf(log_bytes)
            if not narrative or not intervals:
                raise ValueError(f"No parseable source description: {core_id}")
            interval_tops_source_order = [segment["top_cm"] for segment in intervals]
            intervals.sort(key=lambda segment: segment["top_cm"])
            with Image.open(io.BytesIO(photo_bytes)) as opened:
                photo_size = list(opened.size)
            log_path = REVIEW / "logs" / f"{core_id}.pdf"
            photo_path = REVIEW / "photos" / f"{core_id}.jpg"
            log_path.write_bytes(log_bytes)
            photo_path.write_bytes(photo_bytes)
            contiguous = all(abs(a["base_cm"] - b["top_cm"]) < 0.11
                             for a, b in zip(intervals, intervals[1:]))
            covered = (abs(intervals[0]["top_cm"]) < 0.11
                       and core_length is not None
                       and abs(intervals[-1]["base_cm"] - core_length) < 0.11
                       and contiguous)
            records.append({
                "source_id": "usgs_ds850_chandeleur_2012",
                "source_url": OFFICIAL,
                "core_id": core_id,
                "log_zip_entry": log_names[core_id],
                "photo_zip_entry": photo_names[core_id],
                "xray_zip_entries": sorted(xray_names[core_id]),
                "log_pdf": str(log_path.relative_to(ROOT)),
                "photo": str(photo_path.relative_to(ROOT)),
                "photo_size_px": photo_size,
                "log_pages": pages,
                "core_length_cm": core_length,
                "description_en": narrative,
                "intervals": intervals,
                "interval_tops_source_text_order": interval_tops_source_order,
                "intervals_cover_core_length": covered,
                "pairing_basis": "Exact source core ID in official photo filename and official PDF log filename; source PDF also embeds the core photograph.",
                "pixel_to_depth_alignment": "not_verified",
                "russian_translation_status": "pending",
                "training_approved": False,
            })

    jsonl = SOURCE / "source_direct_pairs_quarantine.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "source_id": "usgs_ds850_chandeleur_2012",
        "source_url": OFFICIAL,
        "report_title": "Sediment data collected in 2012 from the northern Chandeleur Islands, Louisiana",
        "archive_files": {kind: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                  "sha256": sha256(path), "zip_crc_error": crc[kind]}
                          for kind, path in ARCHIVES.items()},
        "core_photo_log_pairs": len(records),
        "source_direct_intervals": sum(len(row["intervals"]) for row in records),
        "fully_covered_core_logs": sum(row["intervals_cover_core_length"] for row in records),
        "cores_with_xrays": sum(bool(row["xray_zip_entries"]) for row in records),
        "training_approved_pairs": 0,
        "reason_not_trained": "English source descriptions need Russian review; depth-to-pixel mapping and photo observability of each described feature have not been checked.",
        "records": str(jsonl.relative_to(ROOT)),
        "review_html": str((REVIEW / "index.html").relative_to(ROOT)),
    }
    (SOURCE / "inventory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cards = []
    for row in records:
        length_label = (f"{row['core_length_cm']:g} см" if row["core_length_cm"] is not None
                        else "длина не извлечена")
        interval_items = "".join(
            f"<li>{segment['top_cm']:g}–{segment['base_cm']:g} см: {html.escape(segment['description_en'])}</li>"
            for segment in row["intervals"]
        )
        cards.append(
            f"<article><h2>{row['core_id']} · {length_label}</h2>"
            f"<p><a href='logs/{row['core_id']}.pdf'>Исходный журнал PDF</a> · "
            f"рентгеновских частей: {len(row['xray_zip_entries'])} · "
            f"непрерывные интервалы: {'да' if row['intervals_cover_core_length'] else 'нет'}</p>"
            f"<img loading='lazy' src='photos/{row['core_id']}.jpg' alt='Фото {row['core_id']}'>"
            f"<div><ol>{interval_items}</ol></div></article>"
        )
    page = ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS DS850 · сверка керна</title>"
            "<style>body{font:16px system-ui;margin:24px;background:#f5f5f3;color:#20252a}"
            "article{display:grid;grid-template-columns:180px minmax(150px,260px) 1fr;gap:18px;"
            "padding:18px;margin:16px 0;background:#fff;border:1px solid #ddd;border-radius:10px}"
            "article h2,article p{grid-column:1/-1;margin:0}img{grid-column:2;max-width:100%;max-height:820px}"
            "article div{grid-column:3}li{margin-bottom:10px}@media(max-width:700px){article{display:block}}"
            "</style><h1>USGS DS850: 31 фото и первичные журналы</h1>"
            "<p>Английский текст извлечён из исходных PDF. Точная привязка интервалов к пикселям и русские переводы ожидают проверки. "
            f"<a href='{OFFICIAL}'>Официальный выпуск</a>.</p>" + "".join(cards) + "</html>")
    (REVIEW / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("core_photo_log_pairs", "source_direct_intervals", "fully_covered_core_logs", "cores_with_xrays", "training_approved_pairs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
