"""Import explicit human decisions for Fire Island ruler-aligned VLM windows.

Only checked records from the review page become training examples. P25-C05
is kept as a different-core holdout for this small pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "data/usgs_fire_island_vertical_pilot"


def load_review(path: Path) -> dict:
    review = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(review, dict) or not isinstance(review.get("records"), list):
        raise ValueError("Expected review JSON exported by the Fire Island review page")
    return review


def record_to_example(candidate: dict, approved: dict, reviewer: str, reviewed_at: str,
                      source_pdf_sha256: str) -> dict:
    top, base = candidate["crop_interval_cm"]
    description = approved["description_ru"].strip()
    top_m, base_m = round(top / 100, 5), round(base / 100, 5)
    crop = PILOT / candidate["core_crop"]
    if not crop.is_file() or hashlib.sha256(crop.read_bytes()).hexdigest() != candidate["core_crop_sha256"]:
        raise ValueError(f"Crop missing or changed: {candidate['id']}")
    split = "heldout" if candidate["core_id"] == "P25-C05" else "train"
    prompt = (f"<image>\nЭто фрагмент керна {candidate['core_id']}, глубина {top:g}–{base:g} см "
              "от верха керна. Опиши по-русски только признаки, различимые на этом фото: "
              "цвет, видимую структуру, прослои и контакты. Не угадывай минералы, "
              "точную зернистость или обстановку осадконакопления. "
              "Верни только JSON с ключом description_segments и одним объектом "
              "top_m, base_m, description.")
    target = {"description_segments": [{"top_m": top_m, "base_m": base_m,
                                        "description": description}]}
    return {
        "image": str(crop.resolve()),
        "conversations": [{"from": "human", "value": prompt},
                          {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)}],
        "metadata": {
            "record_id": candidate["id"], "record_type": "human_approved_source_aligned_visual_window",
            "source_group": "USGS_Fire_Island_2016", "core_id": candidate["core_id"],
            "split": split, "source_url": "https://www.usgs.gov/data/archive-sediment-data-vibracores-collected-2016-fire-island-new-york",
            "license": "CC0 1.0 Universal", "source_pdf_page": candidate["source_pdf_page"],
            "source_pdf_sha256": source_pdf_sha256,
            "source_record_id": candidate["source_record_id"],
            "source_description_en": candidate["source_description_en"],
            "source_interval_cm": candidate["source_interval_cm"],
            "window_cm": candidate["crop_interval_cm"],
            "source_photo_member": candidate["photo_member"],
            "photo_crop_box_px": candidate["crop_box_xyxy_px"],
            "image_sha256": candidate["core_crop_sha256"],
            "reviewer": reviewer, "reviewed_at": reviewed_at,
            "ruler_calibration_visually_checked": True,
            "russian_visual_target_reviewed": True,
            "training_approved": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", type=Path, help="JSON exported by review.html")
    parser.add_argument("--output-dir", type=Path, default=PILOT / "reviewed")
    args = parser.parse_args()
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    review = load_review(args.review_json)
    if review.get("source_pdf_sha256") != manifest["source_pdf_sha256"]:
        raise ValueError("Review belongs to a different source PDF")
    candidates = {record["id"]: record for record in manifest["candidates"]}
    rows = review["records"]
    if len(rows) != len(candidates) or {r.get("id") for r in rows} != set(candidates):
        raise ValueError("Review must contain exactly the seven known candidate IDs")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate candidate ID in review")
    approved_rows = []
    for row in rows:
        candidate = candidates[row["id"]]
        if row.get("core_crop_sha256") != candidate["core_crop_sha256"]:
            raise ValueError(f"Crop hash mismatch for {row['id']}")
        if not isinstance(row.get("approved"), bool):
            raise ValueError(f"Missing explicit approval decision for {row['id']}")
        if row["approved"]:
            description = row.get("description_ru")
            if not isinstance(description, str) or len(description.strip()) < 15:
                raise ValueError(f"Approved description too short for {row['id']}")
            approved_rows.append((candidate, row))
    reviewer = review.get("reviewer", "").strip()
    reviewed_at = review.get("reviewed_at", "")
    if approved_rows and (not reviewer or not isinstance(reviewed_at, str) or not reviewed_at):
        raise ValueError("Approved examples require reviewer name and review timestamp")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = [record_to_example(c, r, reviewer, reviewed_at, manifest["source_pdf_sha256"])
                for c, r in approved_rows]
    for split in ("train", "heldout"):
        output = args.output_dir / f"{split}.jsonl"
        subset = [example for example in examples if example["metadata"]["split"] == split]
        output.write_text("".join(json.dumps(example, ensure_ascii=False) + "\n" for example in subset), encoding="utf-8")
    summary = {"review_file": str(args.review_json.resolve()), "reviewer": reviewer,
               "total_candidates": len(candidates), "approved": len(examples),
               "train": sum(e["metadata"]["split"] == "train" for e in examples),
               "heldout": sum(e["metadata"]["split"] == "heldout" for e in examples),
               "warning": "A seven-window, two-core pilot cannot establish production accuracy; hold out P25-C05 entirely."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
