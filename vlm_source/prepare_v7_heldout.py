"""Make a photo-disjoint R-41 split for an honest within-well VLM experiment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "e_drive_interval_crops_v2" / "qwen3_vl_interval_crops.jsonl"
OUT = ROOT / "data" / "v7_heldout"
WINDOWS = [(3240.0, 3266.0), (3840.0, 3857.0)]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def overlaps(interval: list[float], window: tuple[float, float]) -> bool:
    return interval[0] < window[1] and interval[1] > window[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = rows(SOURCE)
    visual = [r for r in source if r.get("metadata", {}).get("record_type") == "user_confirmed_interval_crop"]
    text_only = [r for r in source if r.get("metadata", {}).get("record_type") == "text_only_knowledge"]
    if len(visual) != 181 or any(r["metadata"].get("training_approved") is not True for r in visual):
        raise ValueError("The approved source manifest has changed; inspect the split before continuing.")
    heldout_photos = {
        r["metadata"]["source_photo_record_id"]
        for r in visual
        if any(overlaps(r["metadata"]["source_description_interval_m"], window) for window in WINDOWS)
    }
    train = [r for r in visual if r["metadata"]["source_photo_record_id"] not in heldout_photos]
    test = [r for r in visual if r["metadata"]["source_photo_record_id"] in heldout_photos]
    train_photos = {r["metadata"]["source_photo_record_id"] for r in train}
    test_photos = {r["metadata"]["source_photo_record_id"] for r in test}
    train_rows = {r["metadata"]["source_row"] for r in train}
    test_rows = {r["metadata"]["source_row"] for r in test}
    if train_photos & test_photos or train_rows & test_rows:
        raise ValueError("A source photo or description row leaked across the split.")
    write(OUT / "train_visual.jsonl", train)
    write(OUT / "train_mixed.jsonl", train + text_only)
    write(OUT / "test_visual.jsonl", test)
    summary = {
        "split_type": "within_well_photo_disjoint_stratigraphic_blocks",
        "source_well": "R-41",
        "heldout_depth_windows_m": WINDOWS,
        "train_visual_crops": len(train), "train_unique_photos": len(train_photos),
        "train_unique_description_rows": len(train_rows),
        "train_text_only": len(text_only),
        "test_visual_crops": len(test), "test_unique_photos": len(test_photos),
        "test_unique_description_rows": len(test_rows),
        "photo_overlap": 0, "description_row_overlap": 0,
        "test_rows_by_depth_10m": dict(sorted(Counter(str(int(r["metadata"]["source_description_interval_m"][0] // 10) * 10) for r in test).items())),
        "limitation": "Train/test are distinct source photos and description rows but share one well, field, imaging workflow and report; this is not external-well validation.",
        "v5_comparison_warning": "The existing v5 adapter was trained on all these images, so its output on this test split is in-sample and cannot be compared as a held-out score.",
    }
    (OUT / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
