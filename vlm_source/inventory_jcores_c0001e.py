"""Inventory public J-CORES 315/C0001E VCD rows and split-section scans.

The section-depth overlap is machine-calculated for review only. It does not
prove that the whole VCD interval is visible, nor license deployment training.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "external_raw" / "jamstec_jcores_315_c0001e"
VCD = SOURCE / "vcd.zip"
SCANS = SOURCE / "split-section-image.zip"
SECTIONS = SOURCE / "bulk-section.csv"
REVIEW = SOURCE / "review"
OFFICIAL = "https://www.jamstec.go.jp/sio7/j-cores.data/315/C0001E/"
IMAGE_KEY = "split section image::line scanned image::file"
TOP = "Top Core depth (below sea floor) [m CSF-A]"
BOTTOM = "Bottom Core depth (below sea floor) [m CSF-A]"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_csv(archive: zipfile.ZipFile, name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))


def main() -> None:
    with zipfile.ZipFile(VCD) as vcd_zip, zipfile.ZipFile(SCANS) as scan_zip:
        crc = {"vcd": vcd_zip.testzip(), "scans": scan_zip.testzip()}
        if any(crc.values()):
            raise ValueError(f"ZIP CRC errors: {crc}")
        vcd = zip_csv(vcd_zip, "bulk-lithology-description.csv")
        scans = zip_csv(scan_zip, "bulk-split-section-image.csv")
        if len(scans) != 99 or len(vcd) != 547:
            raise ValueError(f"Unexpected source row counts: scans={len(scans)}, vcd={len(vcd)}")
        scan_names = set(scan_zip.namelist())
        REVIEW.mkdir(exist_ok=True)
        (REVIEW / "photos").mkdir(exist_ok=True)
        records = []
        class_counts: Counter[str] = Counter()
        for scan in scans:
            image_name = scan[IMAGE_KEY]
            if image_name not in scan_names:
                raise ValueError(f"Missing official scan: {image_name}")
            top, bottom = float(scan[TOP]), float(scan[BOTTOM])
            if not math.isfinite(top) or not math.isfinite(bottom) or bottom < top:
                raise ValueError(f"Invalid scan interval: {image_name}")
            section = re.search(r"C0001E-\d+[A-Z]+-(?:\d+|CC)", image_name)
            if section is None:
                raise ValueError(f"Missing section ID: {image_name}")
            image_bytes = scan_zip.read(image_name)
            filename = section.group() + ".jpg"
            photo = REVIEW / "photos" / filename
            photo.write_bytes(image_bytes)
            with Image.open(io.BytesIO(image_bytes)) as image:
                dimensions = list(image.size)
            overlaps = []
            for index, row in enumerate(vcd):
                if not row[TOP] or not row[BOTTOM]:
                    continue
                row_top, row_bottom = float(row[TOP]), float(row[BOTTOM])
                overlap_m = max(0.0, min(bottom, row_bottom) - max(top, row_top))
                if overlap_m <= 0:
                    continue
                overlaps.append({
                    "vcd_source_row": index + 2,
                    "top_csf_a_m": row_top,
                    "bottom_csf_a_m": row_bottom,
                    "overlap_m": round(overlap_m, 4),
                    "lithology_name_en": row["VCD lithology name"],
                    "lithology_comment_en": row["VCD lithology comment"],
                    "microscopic_description_en": row["VCD microscopic lithology description::lithology description::text"],
                    "fully_inside_image_depth_range": row_top >= top and row_bottom <= bottom,
                })
            for entry in overlaps:
                if entry["lithology_name_en"]:
                    class_counts[entry["lithology_name_en"]] += 1
            records.append({
                "source_id": "jamstec_jcores_315_c0001e",
                "source_url": OFFICIAL,
                "section_id": section.group(),
                "image_zip_entry": image_name,
                "photo": str(photo.relative_to(ROOT)),
                "image_size_px": dimensions,
                "top_csf_a_m": top,
                "bottom_csf_a_m": bottom,
                "vcd_overlapping_rows": overlaps,
                "alignment_basis": "Numerical CSF-A depth overlap between official split-section image CSV and official VCD CSV.",
                "alignment_status": "candidate_numeric_overlap_only",
                "russian_translation_status": "pending",
                "rights_for_application_training": "unverified",
                "training_approved": False,
            })

    if not SECTIONS.is_file():
        raise FileNotFoundError(SECTIONS)
    target = SOURCE / "section_vcd_candidates_quarantine.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "source_id": "jamstec_jcores_315_c0001e",
        "source_url": OFFICIAL,
        "archives": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in (VCD, SCANS, SECTIONS)],
        "zip_crc_error": crc,
        "vcd_lithology_rows": len(vcd),
        "section_images": len(records),
        "sections_with_vcd_depth_overlap": sum(bool(row["vcd_overlapping_rows"]) for row in records),
        "vcd_rows_with_comment": sum(bool(row["VCD lithology comment"].strip()) for row in vcd),
        "unique_lithology_names": len({row["VCD lithology name"] for row in vcd if row["VCD lithology name"]}),
        "most_common_overlapping_lithologies": class_counts.most_common(12),
        "training_approved_pairs": 0,
        "reason_not_trained": "Numeric section-depth overlap is only a candidate alignment; visual calibration, Russian geological review and application-use rights are pending.",
        "records": str(target.relative_to(ROOT)),
    }
    (SOURCE / "inventory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for row in records:
        section = row["section_id"]
        labels = ", ".join(dict.fromkeys(x["lithology_name_en"] for x in row["vcd_overlapping_rows"] if x["lithology_name_en"])) or "нет пересечения"
        photo = html.escape(Path(row["photo"]).name)
        cards.append(f'<article><a href="photos/{photo}"><img loading="lazy" src="photos/{photo}" alt="{section}"></a>'
                     f'<h2>{section}</h2><p>{row["top_csf_a_m"]}–{row["bottom_csf_a_m"]} m CSF-A</p>'
                     f'<p>{html.escape(labels)}</p><p>Записей VCD: {len(row["vcd_overlapping_rows"])}</p></article>')
    (REVIEW / "index.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>J-CORES C0001E</title>'
        '<style>body{font:15px system-ui;margin:2rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem}article{border:1px solid #bbb;padding:.7rem}img{max-width:100%;height:300px;object-fit:contain;background:#222}</style>'
        '<h1>J-CORES C0001E: секционные фото и VCD</h1><p>Числовое пересечение глубин — кандидат для проверки. Обучение не разрешено.</p><main>'
        + "".join(cards) + "</main></html>", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("section_images", "vcd_lithology_rows", "sections_with_vcd_depth_overlap", "unique_lithology_names", "training_approved_pairs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
