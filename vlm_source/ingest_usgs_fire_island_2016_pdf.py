"""Extract exact source log intervals from the official Fire Island PDF.

Requires pypdf. The PDF's selectable text is used directly; these are source
descriptions, not image-visible Russian VLM training targets.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_fire_island_2016"
PDF = ROOT / "2016-322-FA_CoreLogs.pdf"
INDEX = ROOT / "core_pair_index.json"
SOURCE = "https://www.usgs.gov/data/archive-sediment-data-vibracores-collected-2016-fire-island-new-york"
INTERVAL = re.compile(r"(?m)^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*cm:\s*")
CORE_ID = re.compile(r"Core ID:\s*(P\d+-C\d+)")
FOOTER = re.compile(r"\n(?=\d+(?:\.\d+)?\s*$|Core ID:|Latitude:|LithologySample|Beach Profile)", re.M)


def tidy(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines()]
    return re.sub(r"\s+", " ", " ".join(line for line in lines if line)).strip()


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))["core_pairs"]
    reader = PdfReader(str(PDF))
    if len(reader.pages) != 30:
        raise ValueError(f"Expected 30 PDF pages, found {len(reader.pages)}")
    records = []
    core_summary = {}
    for zero_based in range(3, 30, 2):
        page_no = zero_based + 1
        source_text = reader.pages[zero_based].extract_text() or ""
        id_match = CORE_ID.search(source_text)
        if not id_match or id_match.group(1) not in index:
            raise ValueError(f"Unknown PDF Core ID on page {page_no}")
        core_id = id_match.group(1)
        matches = list(INTERVAL.finditer(source_text))
        if not matches:
            raise ValueError(f"No intervals on page {page_no}")
        core_records = []
        for ordinal, match in enumerate(matches, 1):
            top, base = float(match.group(1)), float(match.group(2))
            if base <= top:
                raise ValueError(f"Invalid interval {core_id}: {top}-{base}")
            end = matches[ordinal].start() if ordinal < len(matches) else len(source_text)
            body = source_text[match.end():end]
            if ordinal == len(matches):
                body = FOOTER.split(body, maxsplit=1)[0]
            description = tidy(body)
            if not description or not description.startswith("SAND"):
                raise ValueError(f"Unexpected source description {core_id} {top}-{base}: {description[:80]}")
            record = {"record_id": f"fire_island_2016_{core_id}_{ordinal:02d}",
                      "core_id": core_id, "source_interval_cm": [top, base],
                      "source_description_en": description,
                      "source_pdf_page": page_no,
                      "source_pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
                      "source_log_member": index[core_id]["log_member"],
                      "source_photo_member": index[core_id]["photo_member"],
                      "source_url": SOURCE, "license": "CC0 1.0 Universal",
                      "text_extraction_method": "selectable text layer of official PDF; no OCR",
                      "photo_interval_aligned": False,
                      "russian_visual_target_reviewed": False,
                      "training_approved": False}
            records.append(record)
            core_records.append(record)
        if core_records[0]["source_interval_cm"][0] != 0:
            raise ValueError(f"Core does not start at zero: {core_id}")
        for left, right in zip(core_records, core_records[1:]):
            if abs(left["source_interval_cm"][1] - right["source_interval_cm"][0]) > 1e-6:
                raise ValueError(f"Gap/overlap in {core_id}: {left['source_interval_cm']} -> {right['source_interval_cm']}")
        core_summary[core_id] = {"pdf_page": page_no,
                                 "intervals": len(core_records),
                                 "last_interval_base_cm": core_records[-1]["source_interval_cm"][1],
                                 "source_photo_member": index[core_id]["photo_member"]}
    if len(core_summary) != 14 or len(records) != 60:
        raise ValueError(f"Expected 14 cores/60 intervals, got {len(core_summary)}/{len(records)}")
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    result = {"source_url": SOURCE, "doi": "10.5066/F7FN15GX",
              "license": "CC0 1.0 Universal",
              "source_pdf_url": "https://coastal.er.usgs.gov/data-release/doi-F7FN15GX/data/2016-322-FA_CoreLogs.pdf",
              "source_pdf_sha256": digest,
              "source_pdf_bytes": PDF.stat().st_size,
              "source_pdf_pages": len(reader.pages),
              "text_extraction_method": "pypdf on selectable PDF text layer, no OCR",
              "core_summary": core_summary, "intervals": records,
              "interval_count": len(records), "training_approved_count": 0,
              "validation": "14 IDs mapped to source photo/log pairs, 60 nonempty SAND descriptions, interval bounds contiguous per core"}
    (ROOT / "source_intervals.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    blocks = []
    for core_id, summary in core_summary.items():
        intervals = [r for r in records if r["core_id"] == core_id]
        photo = f"review_previews/photo_{index[core_id]['photo_member']}"
        log = f"review_previews/log_{index[core_id]['log_member']}"
        lis = "".join(f"<li><b>{r['source_interval_cm'][0]:g}–{r['source_interval_cm'][1]:g} см</b> "
                      f"{html.escape(r['source_description_en'])}</li>" for r in intervals)
        blocks.append(f"<details><summary><b>{html.escape(core_id)}</b> — {len(intervals)} интервалов, "
                      f"PDF с. {summary['pdf_page']}</summary><div class='images'>"
                      f"<a href='{html.escape(photo)}'><img loading='lazy' src='{html.escape(photo)}'></a>"
                      f"<a href='{html.escape(log)}'><img loading='lazy' src='{html.escape(log)}'></a>"
                      f"</div><ol>{lis}</ol></details>")
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>Fire Island: интервалы из PDF</title>
<style>body{font:15px system-ui;max-width:1600px;margin:2rem auto;background:#f7f9fb;color:#20313d}details{background:white;border:1px solid #ccd;border-radius:.6rem;padding:1rem;margin:1rem 0}.images{display:flex;gap:1rem;overflow:auto}.images a{min-width:40%}.images img{max-width:100%;max-height:750px}li{margin:.5rem 0;line-height:1.5}</style>
<h1>USGS Fire Island 2016: 60 исходных интервалов</h1><p>Текст извлечён из текстового слоя официального PDF без OCR. Границы внутри каждого керна непрерывны. Фото связаны с журналами по ID, но координаты интервалов на фото ещё не проверены. Русские визуальные цели не утверждены; обучающих примеров: 0.</p>""" + "".join(blocks) + "</html>"
    (ROOT / "interval_review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"cores": len(core_summary), "intervals": len(records),
                      "source_descriptions_start": dict(Counter(r["source_description_en"].split(",", 1)[0]
                                                               for r in records).most_common(5)),
                      "training_approved": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
