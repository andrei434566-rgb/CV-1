"""Index the official VC4 photo/sketch PDF against its published CSV records.

The resulting page-level joins are for review, not image/interval training labels.
The source CSV includes point contacts, overlapping spans and one wrong segment ID.
Run with the bundled Python that has pypdf and Pillow installed.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_conconully_2025"
SOURCE = ROOT / "source_descriptions.json"
PDF = ROOT / "VC4_sediment_core_photo_sketch_description.pdf"
OUT = ROOT / "page_index.json"
PHOTO_DIR = ROOT / "review_photos"
SEGMENT = re.compile(r"Core Segment:\s*(\d+)\s*-\s*(\d+)\s*cm")

# On each page the selected embedded object is the photographed core, not the
# hand-drawn sketch. These objects were checked against the six page previews.
PHOTO_OBJECTS = ("Im0.jpg", "Im3.jp2", "Im0.jp2", "Im0.jp2", "Im1.jp2", "Im0.jp2")
EXPECTED_SEGMENTS = ((0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 323))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def touches_page(record: dict, top: int, base: int) -> bool:
    a, b = record["source_depth_cm"]
    if a == b:
        return top <= a <= base
    return a < base and b > top


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    pdf_hash = file_sha256(PDF)
    if source["photo_pdf"] is None or source["photo_pdf"]["sha256"] != pdf_hash:
        raise ValueError("PDF hash is absent from or differs from source_descriptions.json")
    reader = PdfReader(PDF)
    if len(reader.pages) != len(EXPECTED_SEGMENTS):
        raise ValueError(f"Expected six VC4 pages, found {len(reader.pages)}")
    PHOTO_DIR.mkdir(exist_ok=True)
    pages = []
    seen_ids: set[str] = set()
    for idx, (page, expected, image_name) in enumerate(zip(reader.pages, EXPECTED_SEGMENTS, PHOTO_OBJECTS), 1):
        page_text = page.extract_text() or ""
        segment = SEGMENT.search(page_text)
        if not segment or tuple(map(int, segment.groups())) != expected or "Core ID: VC4" not in page_text:
            raise ValueError(f"Unexpected header on PDF page {idx}")
        available = {image.name: image for image in page.images}
        if image_name not in available:
            raise ValueError(f"Photo object {image_name} missing on page {idx}")
        with Image.open(io.BytesIO(available[image_name].data)) as opened:
            photo = opened.convert("RGB")
            photo_file = PHOTO_DIR / f"VC4_{expected[0]:03d}_{expected[1]:03d}.jpg"
            photo.save(photo_file, quality=95, subsampling=0)
            width, height = photo.size
        rows = [record for record in source["records"] if touches_page(record, *expected)]
        seen_ids.update(record["record_id"] for record in rows)
        pages.append({
            "pdf_page": idx,
            "core_id": "VC4",
            "segment_cm": list(expected),
            "embedded_photo_object": image_name,
            "photo_only_file": str(photo_file.relative_to(ROOT)).replace("\\", "/"),
            "photo_only_size_px": [width, height],
            "photo_only_sha256": file_sha256(photo_file),
            "pdf_page_text_en": page_text,
            "pdf_page_text_sha256": sha256_bytes(page_text.encode("utf-8")),
            "csv_record_ids_by_numeric_depth": [record["record_id"] for record in rows],
            "point_observation_ids": [record["record_id"] for record in rows if record["source_record_kind"] == "point_observation"],
            "misnamed_segment_record_ids": [record["record_id"] for record in rows if not record["depth_inside_named_segment"]],
            "interval_pixel_aligned": False,
            "russian_visual_target_reviewed": False,
            "training_approved": False,
        })
    unmatched = sorted({r["record_id"] for r in source["records"]} - seen_ids)
    result = {
        "source_url": source["source_url"], "doi": source["doi"], "license": source["license"],
        "source_pdf": PDF.name, "source_pdf_sha256": pdf_hash,
        "source_csv": source["source_csv"], "source_csv_sha256": source["source_csv_sha256"],
        "pages": pages,
        "summary": {"photo_sketch_pages": len(pages), "source_records": len(source["records"]),
                    "source_records_with_numeric_page_overlap": len(seen_ids),
                    "unmatched_source_record_ids": unmatched, "approved_training_pairs": 0},
        "warning": "Page-level numeric overlap is not pixel-depth alignment or proof that each logged feature is visually identifiable on the photographed core. Do not train on these joins directly.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    by_id = {record["record_id"]: record for record in source["records"]}
    for page in pages:
        top, base = page["segment_cm"]
        csv_rows = []
        for rid in page["csv_record_ids_by_numeric_depth"]:
            record = by_id[rid]
            a, b = record["source_depth_cm"]
            flags = []
            if a == b:
                flags.append("точечное наблюдение")
            if not record["depth_inside_named_segment"]:
                flags.append("ошибка ID сегмента в CSV")
            csv_rows.append("<tr><td>{}</td><td>{:g}–{:g}</td><td>{}</td><td>{}</td></tr>".format(
                record["source_csv_row"], a, b, html.escape(", ".join(flags)),
                html.escape(record["source_description_en"])))
        cards.append(f"<section><h2>VC4 {top}–{base} см · PDF стр. {page['pdf_page']}</h2>"
                     f"<div class='grid'><div><h3>Фото без схемы</h3><img class='core' src='{page['photo_only_file']}'></div>"
                     f"<div><h3>Оригинальная страница</h3><img class='page' src='preview/page-{page['pdf_page']}.jpg'></div></div>"
                     "<p>Совпадения по числовой глубине; точный участок фото ещё не выверен. Обучающих пар: 0.</p>"
                     "<table><tr><th>Строка CSV</th><th>Глубина, см</th><th>Флаг</th><th>Исходное описание</th></tr>"
                     + "".join(csv_rows) + "</table><details><summary>Текст описаний со страницы PDF</summary><pre>"
                     + html.escape(page["pdf_page_text_en"]) + "</pre></details></section>")
    html_doc = ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS VC4 — страницы и описания</title>"
                "<style>body{font:15px system-ui;color:#223;background:#f6f7fa;max-width:1550px;margin:2rem auto}"
                "section{background:white;padding:1.2rem;margin:1.5rem 0;border:1px solid #cbd5e1}"
                ".grid{display:flex;gap:2rem;align-items:flex-start}.core{height:760px;max-width:380px}.page{height:760px}"
                "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:.4rem;text-align:left;vertical-align:top}"
                "td:last-child{min-width:500px}pre{white-space:pre-wrap}</style>"
                "<h1>USGS Conconully VC4: фото, страницы журнала и исходные описания</h1>"
                "<p>Источник: DOI 10.5066/P14DXYOD, CC0. Строки CSV связаны со страницами только по числовой глубине. "
                "Один ID сегмента ошибочен, часть интервалов перекрывается; связь каждой строки с пикселями не подтверждена.</p>"
                + "".join(cards) + "</html>")
    (ROOT / "page_review.html").write_text(html_doc, encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
