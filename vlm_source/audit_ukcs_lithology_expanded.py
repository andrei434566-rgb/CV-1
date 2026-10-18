"""Verify every expanded UKCS crop against source row labels and well split."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from inspect_ukcs_labels import ARCHIVE, ROOT


DATA = ROOT / "data/ukcs_q204_q205_lithology_expanded"
OLD = ROOT / "data/ukcs_q204_q205_lithology_candidates/holdout_pilot.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    train, heldout, old_holdout = read(DATA / "train_pilot.jsonl"), read(DATA / "holdout_pilot.jsonl"), read(OLD)
    train_wells = {row["metadata"]["well_id"] for row in train}
    heldout_wells = {row["metadata"]["well_id"] for row in heldout}
    old_holdout_wells = {row["metadata"]["well_id"] for row in old_holdout}
    if train_wells & heldout_wells or train_wells & old_holdout_wells:
        raise ValueError("A holdout well leaked into training")
    if heldout_wells != old_holdout_wells:
        raise ValueError("New and original holdout wells differ")
    all_rows = train + heldout
    by_member = {}
    for row in all_rows:
        meta = row["metadata"]
        by_member.setdefault(meta["label_member"], []).append(meta)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name, items in by_member.items():
            with archive.open(name) as handle:
                labels = np.load(handle, allow_pickle=False)
            depth_name = name.replace("_labels.npy", "_depth.npy")
            with archive.open(depth_name) as handle:
                depth = np.load(handle, allow_pickle=False)
            for item in items:
                top, bottom = item["row_top"], item["row_bottom"]
                if bottom - top != 600 or not np.all(labels[top:bottom] == item["source_label_code"].encode("ascii")):
                    raise ValueError(f"Source label mismatch: {item['record_id']}")
                d = np.diff(depth[top:bottom])
                if np.any(d < 0) or np.any(d > 0.01):
                    raise ValueError(f"Depth discontinuity: {item['record_id']}")
    train_hashes = [digest(row["image"]) for row in train]
    holdout_hashes = {digest(row["image"]) for row in old_holdout}
    if len(set(train_hashes)) != len(train_hashes) or set(train_hashes) & holdout_hashes:
        raise ValueError("Exact image content duplicates or holdout leak")
    report = {
        "source_archive": str(ARCHIVE),
        "train_crops": len(train), "train_wells": sorted(train_wells),
        "train_by_source_class": dict(Counter(row["metadata"]["source_label_code"] for row in train)),
        "new_holdout_crops": len(heldout), "original_fixed_holdout_crops": len(old_holdout),
        "holdout_wells": sorted(old_holdout_wells),
        "source_label_rows_verified": len(all_rows),
        "train_image_content_unique": len(set(train_hashes)),
        "train_vs_original_holdout_duplicate_count": 0,
        "geologist_approved": 0,
        "status": "source-array-aligned experimental candidates; Russian label mapping and visual crop quality need geologist review",
    }
    (DATA / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
