"""Inventory the direct USGS Nauset Marsh descriptions without inventing photo pairs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data/external_raw/usgs_nauset_marsh"
ARCHIVE = DEST / "corelog_csv.zip"
PHOTO_ARCHIVE = DEST / "coreimages.zip"
SOURCE = "https://pubs.usgs.gov/of/2006/1169/data/gis/cores/corelog_csv.zip"
PHOTO_SOURCE = "https://pubs.usgs.gov/of/2006/1169/data/gis/cores/coreimages.zip"


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source ZIP failed CRC check")
        data = archive.read("corelog_csv.csv")
    rows = list(csv.DictReader(io.StringIO(data.decode("latin1"))))
    required = {"Site", "waterdepth_m", "interval_cmbsf", "description", "dep_environ"}
    if len(rows) != 42 or not rows or set(rows[0]) != required:
        raise ValueError("Unexpected source CSV structure")
    converted = []
    for index, row in enumerate(rows, 1):
        if not re.fullmatch(r"SP-2006-\d{2}", row["Site"]):
            raise ValueError(f"Unexpected site on row {index}")
        if not row["description"].strip() or not row["dep_environ"].strip():
            raise ValueError(f"Missing source description or environment on row {index}")
        converted.append({
            "source_id": "usgs_nauset_marsh_2006",
            "source_url": SOURCE,
            "source_row_number": index + 1,
            "site": row["Site"],
            "waterdepth_m_original": row["waterdepth_m"] or None,
            "interval_cmbsf_original": row["interval_cmbsf"] or None,
            "description_original_en": row["description"].strip(),
            "depositional_environment_original_en": row["dep_environ"].strip(),
            "photo_alignment": "unverified",
            "training_approved": False,
            "russian_translation_reviewed": False,
        })
    manifest = DEST / "source_log_rows_quarantine.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
                        encoding="utf-8")
    photos = {}
    if PHOTO_ARCHIVE.exists():
        with zipfile.ZipFile(PHOTO_ARCHIVE) as archive:
            if archive.testzip() is not None:
                raise ValueError("Photo ZIP failed CRC check")
            for number in (2, 3, 4, 6):
                name = f"core{number}.jpg"
                payload = archive.read(name)
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                with Image.open(io.BytesIO(payload)) as image:
                    dimensions = list(image.size)
                photos[f"SP-2006-{number:02d}"] = {
                    "archive_member": name, "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "dimensions_px": dimensions,
                    "image_path": str((DEST / "full_images" / name).resolve()),
                }
    report = {
        "source_zip": SOURCE,
        "catalog": "https://pubs.usgs.gov/of/2006/1169/htmldocs/catalog.htm",
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "zip_crc": "passed",
        "photo_zip": PHOTO_SOURCE if PHOTO_ARCHIVE.exists() else None,
        "photo_archive_bytes": PHOTO_ARCHIVE.stat().st_size if PHOTO_ARCHIVE.exists() else None,
        "photo_archive_sha256": hashlib.sha256(PHOTO_ARCHIVE.read_bytes()).hexdigest()
                                if PHOTO_ARCHIVE.exists() else None,
        "photo_zip_crc": "passed" if PHOTO_ARCHIVE.exists() else None,
        "photo_composites": photos,
        "source_rows": len(converted),
        "sites": dict(Counter(row["site"] for row in converted)),
        "depositional_environment_source_distribution": dict(Counter(
            row["depositional_environment_original_en"] for row in converted)),
        "rows_with_explicit_interval_field": sum(row["interval_cmbsf_original"] is not None
                                                 for row in converted),
        "rows_without_explicit_interval_field": sum(row["interval_cmbsf_original"] is None
                                                    for row in converted),
        "photo_aligned_rows": 0,
        "training_approved_rows": 0,
        "notes": [
            "The table includes nested annotations whose interval field is blank; do not infer a single interval from adjacent rows.",
            "The source depositional environment is an interpretation, not a class visually established for every crop.",
            "Four composite photos are aligned by site only; precise depth-to-pixel calibration has not been verified.",
        ],
    }
    (DEST / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"rows": len(converted), "sites": report["sites"],
                      "photo_aligned_rows": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
