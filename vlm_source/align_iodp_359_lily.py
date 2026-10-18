"""Audit exact depth overlap between CC BY 4.0 IODP 359 core images and LILY.

The output is a review queue. Carbonate subtype labels may require microscopy and
are never approved for visual training by this automatic join.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
PHOTO = ROOT / "data" / "external_raw" / "iodp_359_photo_pilot"
LILY = ROOT / "data" / "external_raw" / "iodp_lily_v2" / "CleanLITH.zip"
HOLES = ("U1465B", "U1466B", "U1469B")


def zip_rows(archive: zipfile.ZipFile, member: str) -> list[dict]:
    with archive.open(member) as stream:
        return list(csv.DictReader((line.decode("utf-8-sig", errors="replace") for line in stream)))


def union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    start, end = intervals[0]
    for left, right in intervals[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def main() -> None:
    with zipfile.ZipFile(LILY) as archive:
        lith = zip_rows(archive, "CleanLITH/X359_CleanLITH.csv")
    by_core: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in lith:
        key = (row["Site"], row["Hole"], row["Core"], row["Type"])
        by_core[key].append(row)
    results = []
    with zipfile.ZipFile(PHOTO / "data_by_hole.zip") as archive:
        for hole in HOLES:
            images = zip_rows(archive, f"359-{hole}-COREPHOTO.csv")
            with zipfile.ZipFile(PHOTO / f"{hole}.zip") as image_zip:
                members = set(image_zip.namelist())
            for photo in images:
                start, end = float(photo["Top depth CSF-A (m)"]), float(photo["Bottom depth CSF-A (m)"])
                if end <= start:
                    raise RuntimeError(f"Invalid photo depth for {photo['Image filename']}")
                filename = photo["Image filename"]
                if filename not in members:
                    raise RuntimeError(f"Photo metadata lacks an image: {filename}")
                key = (photo["Site"], photo["Hole"], photo["Core"], photo["Type"])
                matches = []
                clipped = []
                length_by_class: dict[str, float] = defaultdict(float)
                for row in by_core.get(key, []):
                    top = float(row["Top depth CSF-A (m)"])
                    bottom = float(row["Bottom depth CSF-A (m)"])
                    left, right = max(start, top), min(end, bottom)
                    if right <= left:
                        continue
                    interval = (left, right)
                    clipped.append(interval)
                    label = row["Simplified Lithology"].strip().lower()
                    length_by_class[label] += right - left
                    matches.append({
                        "top_m": top,
                        "bottom_m": bottom,
                        "overlap_top_m": left,
                        "overlap_bottom_m": right,
                        "full_lithology_en": row["Full Lithology"],
                        "simplified_lithology_en": label,
                        "section": row["Sect"],
                    })
                covered = union_length(clipped)
                summed = sum(right - left for left, right in clipped)
                dominant = max(length_by_class, key=length_by_class.get) if length_by_class else None
                result = {
                    "record_id": f"iodp359_{hole}_{photo['Core']}{photo['Type']}",
                    "expedition": 359,
                    "site": photo["Site"],
                    "hole": photo["Hole"],
                    "core": photo["Core"],
                    "type": photo["Type"],
                    "photo_zip": f"{hole}.zip",
                    "photo_zip_member": filename,
                    "photo_depth_csf_a_m": [start, end],
                    "lithology_covered_fraction": round(covered / (end - start), 5),
                    "overlap_duplicate_fraction": round(max(0.0, summed - covered) / (end - start), 5),
                    "matched_lithology_rows": matches,
                    "lithology_overlap_m": {key: round(value, 5) for key, value in sorted(length_by_class.items())},
                    "dominant_lithology_en": dominant,
                    "dominant_fraction_of_photo": round(length_by_class[dominant] / (end - start), 5) if dominant else 0.0,
                    "metadata_alignment_verified": covered > 0 and summed <= covered + 1e-6,
                    "visual_label_reviewed": False,
                    "russian_target_reviewed": False,
                    "training_approved": False,
                    "source_photo_url": "https://zenodo.org/records/14015049",
                    "source_description_url": "https://zenodo.org/records/10425539",
                }
                results.append(result)
    results.sort(key=lambda row: (row["site"], row["hole"], int(row["core"])))
    path = PHOTO / "lily_alignment_candidates.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")
    summary = {
        "source_photo_record": "https://zenodo.org/records/14015049",
        "source_lithology_record": "https://zenodo.org/records/10425539",
        "license_both": "CC BY 4.0",
        "photo_count": len(results),
        "with_any_depth_overlap": sum(row["matched_lithology_rows"] != [] for row in results),
        "at_least_95_percent_covered": sum(row["lithology_covered_fraction"] >= .95 for row in results),
        "no_overlapping_lithology_rows": sum(row["overlap_duplicate_fraction"] <= 1e-6 for row in results),
        "dominant_at_least_90_percent": sum(row["dominant_fraction_of_photo"] >= .9 for row in results),
        "training_approved": 0,
        "reason": "Automated depth joins do not prove that carbonate subtypes or laboratory-derived labels are visually observable.",
        "candidate_manifest": path.name,
    }
    (PHOTO / "alignment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
