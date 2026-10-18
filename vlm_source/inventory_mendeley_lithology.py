"""Audit the original Mendeley lithology ZIP and re-split by whole well.

Source directory names are direct lithology labels. They are *not* descriptions
or facies labels. This script never extracts or trains on archive contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data/external_raw/mendeley_lithology/b6rj29mrss-1.zip"
OUT = ROOT / "data/external_raw/mendeley_lithology"
CLASSES = {"sandstone", "limestone", "shale", "garbage"}
SPLITS = {"train", "val", "test"}
TEST_WELLS = {"08PD04c", "HODD2", "COD012", "SAE 6"}
VAL_WELLS = {"WC05D001", "MMDD1", "COD013", "IHAD2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-crc", action="store_true", help="Read every member and validate ZIP CRC (slow).")
    args = parser.parse_args()
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    assert not TEST_WELLS & VAL_WELLS
    original_counts = Counter()
    group_counts = Counter()
    well_counts = Counter()
    original_wells = defaultdict(set)
    image_keys = defaultdict(list)
    malformed = []
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_crc_member = archive.testzip() if args.full_crc else None
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            if len(parts) != 4 or parts[0] != "Lithology dataset" or parts[1] not in SPLITS or parts[2] not in CLASSES or not parts[3].lower().endswith((".jpg", ".jpeg", ".png")):
                malformed.append(info.filename)
                continue
            _, original_split, label, filename = parts
            well = filename.split("_", 1)[0].strip()
            if not well:
                malformed.append(info.filename)
                continue
            group_split = "test" if well in TEST_WELLS else "val" if well in VAL_WELLS else "train"
            original_counts[(original_split, label)] += 1
            group_counts[(group_split, label)] += 1
            well_counts[(well, label)] += 1
            original_wells[original_split].add(well)
            key = (info.CRC, info.file_size)
            image_keys[key].append((info.filename, label, well, group_split))
            records.append({
                "archive_member": info.filename,
                "source_split": original_split,
                "source_label": label,
                "source_well_id": well,
                "well_split": group_split,
                "size_bytes": info.file_size,
                "zip_crc32": f"{info.CRC:08x}",
                "label_provenance": "source_direct_folder",
                "alignment_status": "not_applicable_2x2cm_crop",
                "facies_label": None,
                "description_target": None,
            })
    wells = {well for well, _ in well_counts}
    if not (TEST_WELLS | VAL_WELLS) <= wells:
        raise ValueError("Фиксированный список скважин не совпал с архивом")
    duplicate_candidates = [items for items in image_keys.values() if len(items) > 1]
    sha_groups = defaultdict(list)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for group in duplicate_candidates:
            for item in group:
                member = item[0]
                sha_groups[hashlib.sha256(archive.read(member)).hexdigest()].append(item)
    duplicates = [items for items in sha_groups.values() if len(items) > 1]
    cross_label = [items for items in duplicates if len({x[1] for x in items}) > 1]
    cross_well = [items for items in duplicates if len({x[2] for x in items}) > 1]
    cross_new_split = [items for items in duplicates if len({x[3] for x in items}) > 1]
    suspect_members = {item[0] for group in cross_label + cross_well + cross_new_split for item in group}
    for record in records:
        record["quarantine_reason"] = "duplicate_conflict_or_cross_well" if record["archive_member"] in suspect_members else None
        record["eligible_for_lithology_pilot"] = record["quarantine_reason"] is None
    index_path = OUT / "source_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "Alzubaidi Lithology dataset",
        "source_url": "https://data.mendeley.com/datasets/b6rj29mrss/1",
        "doi": "10.17632/b6rj29mrss.1",
        "license": "CC BY 4.0",
        "archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "zip_crc_full_check": "passed" if args.full_crc and bad_crc_member is None else bad_crc_member or "not_run",
        "record_count": len(records),
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:20],
        "wells": len(wells),
        "original_split_wells": {split: sorted(original_wells[split]) for split in sorted(SPLITS)},
        "original_split_overlap_well_count": {
            "train_val": len(original_wells["train"] & original_wells["val"]),
            "train_test": len(original_wells["train"] & original_wells["test"]),
            "val_test": len(original_wells["val"] & original_wells["test"]),
        },
        "new_split_wells": {
            "train": sorted(wells - TEST_WELLS - VAL_WELLS),
            "val": sorted(VAL_WELLS),
            "test": sorted(TEST_WELLS),
        },
        "source_counts": {f"{split}/{label}": original_counts[(split, label)] for split in sorted(SPLITS) for label in sorted(CLASSES)},
        "new_counts": {f"{split}/{label}": group_counts[(split, label)] for split in sorted(SPLITS) for label in sorted(CLASSES)},
        "well_class_counts": {well: {label: well_counts[(well, label)] for label in sorted(CLASSES)} for well in sorted(wells)},
        "duplicate_candidate_crc_size_groups": len(duplicate_candidates),
        "exact_duplicate_sha256_groups": len(duplicates),
        "cross_label_duplicate_groups": len(cross_label),
        "cross_well_duplicate_groups": len(cross_well),
        "cross_new_split_duplicate_groups": len(cross_new_split),
        "quarantined_records": len(suspect_members),
        "duplicate_conflict_examples": [items[:4] for items in (cross_label + cross_well + cross_new_split)[:12]],
        "training_scope": "small-crop lithology and non-core detection only; no facies or free-text description labels",
        "eligibility_note": "Source-folder labels are direct but not individually reviewed. Stage images only after successful decode; quarantine conflicts and duplicates across wells or splits.",
    }
    path = OUT / "inventory.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("archive_bytes", "archive_sha256", "zip_crc_full_check", "record_count", "malformed_count", "wells", "original_split_overlap_well_count", "new_counts", "cross_label_duplicate_groups", "cross_well_duplicate_groups", "cross_new_split_duplicate_groups", "quarantined_records")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
