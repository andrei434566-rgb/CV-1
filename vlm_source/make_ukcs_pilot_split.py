"""Create a whole-well, image-quality-filtered experimental UKCS split."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ukcs_q204_q205_lithology_candidates"
HOLDOUT_WELLS = {"204-19-7", "204-20-3", "204-24a-7"}
MAX_BLACK_FRACTION = 0.15


def black_fraction(path: str) -> float:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    return float((rgb.max(axis=2) < 18).mean())


def main() -> None:
    candidates = [json.loads(line) for line in (DATA / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    train, holdout, excluded = [], [], []
    for row in candidates:
        fraction = black_fraction(row["image"])
        row["metadata"]["black_fraction_after_edge_trim"] = round(fraction, 4)
        if fraction > MAX_BLACK_FRACTION:
            excluded.append({"record_id": row["metadata"]["record_id"], "well_id": row["metadata"]["well_id"], "black_fraction": round(fraction, 4)})
        elif row["metadata"]["well_id"] in HOLDOUT_WELLS:
            holdout.append(row)
        else:
            train.append(row)
    train_wells = {row["metadata"]["well_id"] for row in train}
    holdout_wells = {row["metadata"]["well_id"] for row in holdout}
    if train_wells & holdout_wells or holdout_wells != HOLDOUT_WELLS:
        raise ValueError("Invalid whole-well split")
    for name, rows in (("train_pilot.jsonl", train), ("holdout_pilot.jsonl", holdout)):
        (DATA / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "candidate_total": len(candidates),
        "train_images": len(train),
        "train_wells": sorted(train_wells),
        "train_class_counts": dict(Counter(row["metadata"]["source_label_code"] for row in train)),
        "holdout_images": len(holdout),
        "holdout_wells": sorted(holdout_wells),
        "holdout_class_counts": dict(Counter(row["metadata"]["source_label_code"] for row in holdout)),
        "excluded_high_black": len(excluded),
        "max_black_fraction": MAX_BLACK_FRACTION,
        "excluded_records": excluded,
        "geologist_approved": 0,
        "status": "experimental split only; no claims of full geological accuracy",
    }
    (DATA / "pilot_split_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidate_total", "train_images", "train_wells", "train_class_counts", "holdout_images", "holdout_wells", "holdout_class_counts", "excluded_high_black")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
