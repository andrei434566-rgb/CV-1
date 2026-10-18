"""Join IODP 366 core photos to LILY descriptions at core/depth level.

This is an audit, not a training set: section geometry and visual attributes
still require review before any interval can be used for VLM fine-tuning.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "data" / "external_raw" / "iodp_366_core_photos"
LILY = ROOT / "data" / "external_raw" / "iodp_lily_v2" / "CleanLITH.zip"
OUT = ROOT / "data" / "iodp_366_photo_lily_candidates"


def csv_rows(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))


def key(row: dict[str, str]) -> tuple[str, str, str, int, str]:
    return (
        row["Exp"].strip(), row["Site"].strip(), row["Hole"].strip(),
        int(row["Core"]), row["Type"].strip(),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(LILY) as archive:
        lith = csv_rows(archive.read("CleanLITH/X366_CleanLITH.csv"))
    by_core: dict[tuple[str, str, str, int, str], list[dict[str, str]]] = {}
    for row in lith:
        by_core.setdefault(key(row), []).append(row)

    candidates = []
    with zipfile.ZipFile(PHOTOS / "data_by_hole.zip") as archive:
        for member in archive.namelist():
            for photo in csv_rows(archive.read(member)):
                image_name = photo["Image filename"]
                image_path = PHOTOS / "images" / image_name
                if not image_path.exists():
                    continue
                top = float(photo["Top depth CSF-A (m)"])
                bottom = float(photo["Bottom depth CSF-A (m)"])
                matches = []
                for row in by_core.get(key(photo), []):
                    a = float(row["Top depth CSF-A (m)"])
                    b = float(row["Bottom depth CSF-A (m)"])
                    overlap = max(0.0, min(bottom, b) - max(top, a))
                    if overlap > 0:
                        matches.append({
                            "sample_id": row["SampleID"],
                            "top_m_csf_a": a,
                            "bottom_m_csf_a": b,
                            "overlap_m": round(overlap, 4),
                            "section": row["Sect"],
                            "original_full_lithology": row["Full Lithology"],
                            "simplified_lithology": row["Simplified Lithology"],
                            "lithology_type": row["Lithology Type"],
                        })
                candidates.append({
                    "image": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                    "source_photo_csv": member,
                    "expedition": photo["Exp"],
                    "site": photo["Site"],
                    "hole": photo["Hole"],
                    "core": int(photo["Core"]),
                    "core_type": photo["Type"],
                    "top_m_csf_a": top,
                    "bottom_m_csf_a": bottom,
                    "source_text_id": photo["Text ID"],
                    "lily_intervals": matches,
                    "match_level": "same expedition/site/hole/core/type and overlapping CSF-A depth",
                    "pixel_depth_alignment_reviewed": False,
                    "russian_description_reviewed": False,
                    "training_approved": False,
                })

    counts = Counter(row["simplified_lithology"] for item in candidates for row in item["lily_intervals"])
    payload = {
        "photo_source": "https://zenodo.org/records/19614224",
        "photo_rights": "https://www.iodp.tamu.edu/about/copyright.html",
        "lithology_source": "https://zenodo.org/records/10425539",
        "training_approved": False,
        "reason": "Core/depth identity is exact, but interval-to-pixel alignment and visual interpretation are unreviewed; some LILY lithologies rely on laboratory observations.",
        "summary": {
            "photos": len(candidates),
            "photos_with_lily_intervals": sum(bool(c["lily_intervals"]) for c in candidates),
            "matching_lily_intervals": sum(len(c["lily_intervals"]) for c in candidates),
            "lithology_counts": dict(counts.most_common()),
        },
        "candidates": candidates,
    }
    (OUT / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cards = []
    for item in candidates:
        cid = f"{item['site']}{item['hole']}-{item['core']}{item['core_type']}"
        descriptions = "<br>".join(
            f"{row['top_m_csf_a']:.3f}–{row['bottom_m_csf_a']:.3f} м, "
            f"секция {html.escape(row['section'])}: "
            f"{html.escape(row['original_full_lithology'])}"
            for row in item["lily_intervals"]
        ) or "В LILY нет пересекающихся интервалов"
        image = "../" + item["image"]
        cards.append(
            f"<article><h2>{html.escape(cid)} · {item['top_m_csf_a']:.3f}–{item['bottom_m_csf_a']:.3f} м</h2>"
            f"<img src='{html.escape(image)}' alt='{html.escape(cid)}'><p>{descriptions}</p>"
            "<strong>Только проверка соответствия керна и глубины; в обучение не допущено.</strong></article>"
        )
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>IODP 366 · сверка фото и LILY</title>
<style>body{font:16px system-ui;background:#f4f2ec;color:#282c30;margin:2rem}article{background:white;margin:1rem 0;padding:1rem;border-radius:8px}img{height:480px;width:auto;background:black;float:left;margin:0 1.5rem 1rem 0}article:after{content:'';display:block;clear:both}p{line-height:1.5}strong{color:#9b4a0a}</style>
<h1>IODP 366: первичная сверка фото и литологии</h1><p>Описание LILY относится к глубине; пиксельная привязка не проверена. Оригиналы хранятся рядом.</p>""" + "\n".join(cards) + "</html>"
    (OUT / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
