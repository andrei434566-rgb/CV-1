"""Extract reviewable core photos and page text from official USGS DS1061.

Run with the bundled Python runtime (pypdf and Pillow). No training labels
are generated: graphic lithology boundaries require depth alignment review.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external_raw" / "usgs_cache_creek_ds1061" / "ds1061.pdf"
DEST = ROOT / "data" / "usgs_cache_creek_ds1061_review"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    photos = DEST / "photos"
    photos.mkdir(exist_ok=True)
    reader = PdfReader(SOURCE)
    records = []
    for page_index in range(65, len(reader.pages)):
        page = reader.pages[page_index]
        text = page.extract_text() or ""
        core_match = re.search(r"\bCore:\s*([^\r\n]+)", text)
        section_match = re.search(r"\bSection\s+(\d+)\s+of\s+(\d+)\s*\(([^)]+)\)", text)
        if not core_match or not section_match:
            continue
        images = []
        for embedded in page.images:
            try:
                raw = Image.open(io.BytesIO(embedded.data))
                width, height = raw.size
                if width >= 100 and height >= 500 and height / width >= 4:
                    images.append((width * height, embedded, raw))
            except Exception:
                continue
        if not images:
            continue
        _, embedded, raw = max(images, key=lambda item: item[0])
        target = photos / f"pdf_page_{page_index + 1:03d}.png"
        raw.convert("RGB").save(target, optimize=True)
        records.append({
            "pdf_page": page_index + 1,
            "core_id": core_match.group(1).strip(),
            "borehole_id": core_match.group(1).split("#", 1)[0].strip(),
            "section": int(section_match.group(1)),
            "sections_total": int(section_match.group(2)),
            "source_depth_text": section_match.group(3).strip(),
            "image": str(target.relative_to(ROOT)).replace("\\", "/"),
            "image_size_px": list(raw.size),
            "source_embedded_image": embedded.name,
            "photo_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "page_text": text,
            "depth_pixel_alignment_reviewed": False,
            "interval_labels_reviewed": False,
            "training_approved": False,
        })
        if len(records) % 20 == 0:
            print(f"Extracted {len(records)} section photos", flush=True)

    summary = {
        "source": "https://pubs.usgs.gov/ds/1061/ds1061.pdf",
        "doi": "10.3133/ds1061",
        "source_pdf_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "section_photos": len(records),
        "distinct_core_ids": len({r["core_id"] for r in records}),
        "distinct_borehole_ids": len({r["borehole_id"] for r in records}),
        "core_sections": dict(Counter(r["core_id"] for r in records)),
        "training_approved": False,
        "reason": "Lithology symbols and remarks are page-level; precise interval-to-photo pixel coordinates and Russian visual labels are unreviewed.",
    }
    (DEST / "manifest.json").write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for record in records:
        labels = re.findall(r"\b(?:Sand|Clay|Silt|Sandy silt|Silty sand|Silty clay|Clayey sand|Clayey silt)\b", record["page_text"])
        rel = "photos/" + Path(record["image"]).name
        cards.append(
            f"<article><h2>PDF p. {record['pdf_page']}: {html.escape(record['core_id'])}, "
            f"секция {record['section']} · {html.escape(record['source_depth_text'])}</h2>"
            f"<img src='{html.escape(rel)}' alt='Фото секции'>"
            f"<p>Слова в журнале (без интервальной привязки): {html.escape(', '.join(labels[:30]))}</p>"
            "<p><b>Не допущено в обучение: требуется проверка глубин и визуальных меток.</b></p></article>"
        )
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS DS1061 · снимки секций</title>
<style>body{font:16px system-ui;background:#f3f1ea;margin:2rem;color:#252a2c}article{background:white;padding:1rem;margin:1rem 0;border-radius:8px;min-height:320px}img{height:300px;width:auto;float:left;margin:0 1.5rem 1rem 0}article:after{content:'';display:block;clear:both}b{color:#9b4a0a}</style>
<h1>USGS DS1061: фото и текст страниц</h1><p>Оригинальный PDF хранится в external_raw. Здесь извлечены изображения без интерпретации границ.</p>""" + "\n".join(cards) + "</html>"
    (DEST / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
