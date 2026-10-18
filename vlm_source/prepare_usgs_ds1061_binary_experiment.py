"""Prepare a source-labeled, explicitly experimental DS1061 binary pilot.

The PDF remark-to-image mapping is approximate. No row receives production
approval. Entire source sites 2, 5 and 12 are held out before oversampling.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "usgs_cache_creek_ds1061_windows" / "manifest.json"
OUT = ROOT / "data" / "usgs_cache_creek_ds1061_binary_experiment"
SAND = {"Sand", "Silty sand", "Clayey sand", "Gravelly sand"}
CLAY = {"Clay", "Silty clay", "Sandy clay", "Gravelly silty clay"}
HOLDOUT_SITES = {"2", "5", "12"}
PROMPT = (
    "<image>\nПо фотографии этого фрагмента керна выбери преобладающую укрупнённую "
    "литологическую группу: песчаный или глинистый. "
    "Ответь ровно одним словом: песчаный или глинистый."
)


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))["proposals"]
    by_page: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_page[row["page"]].append(row)
    unique_train = []
    heldout = []
    excluded = Counter()
    for row in rows:
        label = row["source_label_en"]
        if label not in SAND | CLAY:
            excluded["mixed_or_silt_rich_source_label"] += 1
            continue
        if row["crop_y_px"][1] - row["crop_y_px"][0] < 120:
            excluded["short_image"] += 1
            continue
        closest = min(
            (abs(row["source_label_pdf_y_pt"] - other["source_label_pdf_y_pt"])
             for other in by_page[row["page"]] if other is not row),
            default=999,
        )
        if closest < 20:
            excluded["close_to_other_lithology_remark"] += 1
            continue
        site = re.match(r"\d+", row["borehole_id"])
        if not site:
            excluded["missing_site_id"] += 1
            continue
        site_id = site.group()
        target = "песчаный" if label in SAND else "глинистый"
        image = ROOT / row["image"]
        record = {
            "image": str(image),
            "conversations": [
                {"from": "human", "value": PROMPT},
                {"from": "gpt", "value": target},
            ],
            "metadata": {
                "record_id": f"ds1061_p{row['page']:03d}_{Path(row['image']).stem}",
                "record_type": "experimental_weak_pdf_remark_photo_window",
                "source_group": "USGS_DS1061",
                "source_url": "https://pubs.usgs.gov/ds/1061/ds1061.pdf",
                "source_pdf_page": row["page"],
                "site_group": site_id,
                "well_id": row["borehole_id"],
                "core_id": row["core_id"],
                "source_label_en": label,
                "source_label_y_pt": row["source_label_pdf_y_pt"],
                "min_other_label_distance_pt": round(closest, 1),
                "source_crop_y_px": row["crop_y_px"],
                "label_provenance": "source_direct_remark_with_approximate_pixel_window",
                "source_label_translation_ru": target,
                "geology_approved": False,
                "training_approved": False,
            },
        }
        (heldout if site_id in HOLDOUT_SITES else unique_train).append(record)
    train_sites = {r["metadata"]["site_group"] for r in unique_train}
    holdout_sites = {r["metadata"]["site_group"] for r in heldout}
    if train_sites & holdout_sites or holdout_sites != HOLDOUT_SITES:
        raise ValueError("Site-level train/test split failed")
    # Deterministic class rebalancing on training sites only. All counts are
    # reported on the unique rows; no held-out example is duplicated.
    train = []
    for row in unique_train:
        copies = 3 if row["conversations"][1]["value"] == "песчаный" else 1
        for copy in range(copies):
            cloned = json.loads(json.dumps(row, ensure_ascii=False))
            cloned["metadata"]["training_copy_index"] = copy
            train.append(cloned)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in (("train_experimental.jsonl", train), ("heldout.jsonl", heldout)):
        (OUT / name).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in content) + "\n", encoding="utf-8")
    summary = {
        "scope": "Research-only sand-rich versus clay-rich source labels; NOT full lithology, description or depositional facies.",
        "source": "https://pubs.usgs.gov/ds/1061/ds1061.pdf",
        "train_sites": sorted(train_sites, key=int),
        "heldout_sites": sorted(holdout_sites, key=int),
        "unique_train": len(unique_train),
        "train_after_sand_oversampling": len(train),
        "heldout": len(heldout),
        "unique_train_labels": dict(Counter(r["conversations"][1]["value"] for r in unique_train)),
        "train_labels_after_oversampling": dict(Counter(r["conversations"][1]["value"] for r in train)),
        "heldout_labels": dict(Counter(r["conversations"][1]["value"] for r in heldout)),
        "excluded": dict(excluded),
        "training_approved": False,
        "limitations": [
            "PDF remark coordinate is not an interval boundary; photo target remains unreviewed.",
            "Source macroscopic grain size can be difficult or impossible to infer from a 205-pixel-wide photo.",
            "Multiple windows from one core are correlated; heldout split is by entire numbered site.",
            "Sites are all in one settling basin; no ancient-core transfer claim follows.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
