"""Preserve Conconully VC4 field descriptions without inventing clean intervals.

The official CSV mixes true intervals, point contacts, nested observations and
at least one segment-name inconsistency. These are source records, not photo
training labels.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_conconully_2025"
CSV = ROOT / "Conconully_sediment_cores_initial_descriptions.csv"
PDF = ROOT / "VC4_sediment_core_photo_sketch_description.pdf"
SEGMENT = re.compile(r"^(VC\d+)-(\d+)-(\d+)$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with CSV.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 40:
        raise ValueError(f"Expected 40 rows, got {len(rows)}")
    csv_hash = file_sha256(CSV)
    records = []
    for ordinal, row in enumerate(rows, 1):
        match = SEGMENT.fullmatch(row["Core_ID_segment"].strip())
        if not match or match.group(1) != "VC4":
            raise ValueError(f"Unexpected segment ID at row {ordinal}")
        top = float(row["top_depth_cm"])
        base = float(row["bottom_depth_cm"])
        if not (0 <= top <= base <= 323):
            raise ValueError(f"Invalid depth at row {ordinal}: {top}-{base}")
        text = row["Initial_description"].strip()
        if not text:
            raise ValueError(f"Empty description at row {ordinal}")
        segment_range = [float(match.group(2)), float(match.group(3))]
        records.append({
            "record_id": f"conconully_vc4_{ordinal:02d}", "source_csv_row": ordinal + 1,
            "core_id": "VC4", "segment_id_as_published": row["Core_ID_segment"],
            "segment_range_cm_from_id": segment_range,
            "source_depth_cm": [top, base],
            "source_record_kind": "point_observation" if top == base else "interval_or_span",
            "depth_inside_named_segment": segment_range[0] <= top and base <= segment_range[1],
            "source_description_en": text,
            "source_csv_sha256": csv_hash,
            "photo_depth_aligned": False, "russian_visual_target_reviewed": False,
            "training_approved": False,
        })
    intervals = [r for r in records if r["source_record_kind"] == "interval_or_span"]
    points = [r for r in records if r["source_record_kind"] == "point_observation"]
    spans = sorted(intervals, key=lambda r: r["source_depth_cm"])
    overlaps = sum(left["source_depth_cm"][1] > right["source_depth_cm"][0]
                   for left, right in zip(spans, spans[1:]))
    photo_info = None
    if PDF.is_file():
        photo_info = {"name": PDF.name, "bytes": PDF.stat().st_size,
                      "sha256": file_sha256(PDF)}
    result = {
        "source_url": "https://www.usgs.gov/data/sediment-computed-tomography-ct-scans-photographs-and-core-log-descriptions-conconully",
        "doi": "10.5066/P14DXYOD", "license": "CC0 1.0 Universal",
        "source_csv": CSV.name, "source_csv_sha256": csv_hash,
        "photo_pdf": photo_info, "core_id": "VC4", "records": records,
        "summary": {"total": len(records), "nonzero_spans": len(intervals),
                    "point_observations": len(points), "adjacent_span_overlaps": overlaps,
                    "segment_name_depth_mismatches": [r["record_id"] for r in records
                                                       if not r["depth_inside_named_segment"]],
                    "approved_for_training": 0},
        "warning": "Published rows overlap and include zero-length contacts; do not flatten into a continuous interval log or train from them without checking the photo/segment/sketch.",
    }
    (ROOT / "source_descriptions.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table = []
    for r in records:
        top, base = r["source_depth_cm"]
        flag = " ⚠ ID сегмента" if not r["depth_inside_named_segment"] else ""
        table.append(f"<tr><td>{r['source_csv_row']}</td><td>{html.escape(r['segment_id_as_published'])}{flag}</td>"
                     f"<td>{top:g}–{base:g}</td><td>{html.escape(r['source_record_kind'])}</td>"
                     f"<td>{html.escape(r['source_description_en'])}</td></tr>")
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>Conconully VC4: первичные описания</title>
<style>body{font:15px system-ui;max-width:1550px;margin:2rem auto;color:#223;background:#f6f7fa}table{border-collapse:collapse;background:white}th,td{border:1px solid #ccd;padding:.5rem;vertical-align:top}td:last-child{min-width:600px}</style>
<h1>USGS Conconully VC4: первичные описания</h1><p>Исходная таблица содержит интервалы, точечные контакты и вложенные наблюдения. Некорректное имя сегмента сохранено как в источнике и отмечено, без автоматического исправления. Фото пока не привязано к строкам; обучающих пар 0.</p>
<table><tr><th>Строка CSV</th><th>ID сегмента</th><th>Глубина, см</th><th>Тип</th><th>Исходный текст</th></tr>""" + "".join(table) + "</table></html>"
    (ROOT / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
