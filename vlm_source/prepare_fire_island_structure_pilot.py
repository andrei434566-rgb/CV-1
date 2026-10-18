"""Build an experimental Fire Island structure corpus with core-level holdout.

All targets remain unapproved drafts, so train_qwen3vl.py must be invoked with
--experimental-candidates. The split is by entire core, never by image crop.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VERT = ROOT / "data/usgs_fire_island_vertical_pilot"
HORIZ = ROOT / "data/usgs_fire_island_horizontal_pilot"
OUT = ROOT / "data/usgs_fire_island_structure_pilot"
TRAIN_CORES = {"P22-C14", "P10-C03", "P25-C06"}
HELDOUT_CORES = {"P25-C05", "P29-C11"}


def make_example(candidate: dict, root: Path, split: str) -> dict:
    top, base = candidate["crop_interval_cm"]
    image = root / candidate["core_crop"]
    if hashlib.sha256(image.read_bytes()).hexdigest() != candidate["core_crop_sha256"]:
        raise ValueError(f"Image checksum mismatch: {candidate['id']}")
    prompt = (
        f"<image>\nЭто увеличенный фрагмент керна {candidate['core_id']}, участок "
        f"{top:g}–{base:g} см от верха. Кратко опиши по-русски видимый цвет "
        "и осадочную структуру: однородность, пятнистость, слои или их деформацию. "
        "Не угадывай минералы, точную зернистость, сортировку и обстановку "
        "осадконакопления. Верни только JSON с ключом description_segments "
        "и одним объектом top_m, base_m, description."
    )
    target = {"description_segments": [{"top_m": round(top / 100, 5),
                                        "base_m": round(base / 100, 5),
                                        "description": candidate["visible_target_ru_draft"]}]}
    return {
        "image": str(image.resolve()),
        "conversations": [{"from": "human", "value": prompt},
                          {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)}],
        "metadata": {
            "record_id": candidate["id"],
            "record_type": "experimental_source_aligned_visual_structure_window",
            "source_group": "USGS_Fire_Island_2016", "core_id": candidate["core_id"],
            "split": split,
            "source_url": "https://www.usgs.gov/data/archive-sediment-data-vibracores-collected-2016-fire-island-new-york",
            "license": "CC0 1.0 Universal", "source_pdf_page": candidate["source_pdf_page"],
            "source_record_id": candidate["source_record_id"],
            "source_interval_cm": candidate["source_interval_cm"],
            "source_description_en": candidate["source_description_en"],
            "window_cm": candidate["crop_interval_cm"],
            "photo_crop_box_px": candidate.get("crop_box_xyxy_px") or candidate.get("crop_box_xyxy_original_px"),
            "photo_rotation_ccw_deg": candidate.get("rotation_ccw_deg", 0),
            "image_sha256": candidate["core_crop_sha256"],
            "ruler_calibration_visually_checked": True,
            "visual_target_reviewed_by_assistant": True,
            "geology_approved": False,
            "training_approved": False,
        },
    }


def main() -> None:
    candidates = []
    source_pdf_hashes = set()
    for directory in (VERT, HORIZ):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        source_pdf_hashes.add(manifest["source_pdf_sha256"])
        candidates += [(candidate, directory) for candidate in manifest["candidates"]]
    if len(source_pdf_hashes) != 1:
        raise ValueError("Candidates originate from different source PDFs")
    core_ids = {candidate["core_id"] for candidate, _ in candidates}
    if len(candidates) != 21 or core_ids != TRAIN_CORES | HELDOUT_CORES:
        raise ValueError(f"Unexpected pilot composition: {len(candidates)} windows, {core_ids}")
    if len({candidate["id"] for candidate, _ in candidates}) != len(candidates):
        raise ValueError("Duplicate window ID")
    examples = []
    for candidate, directory in candidates:
        split = "train" if candidate["core_id"] in TRAIN_CORES else "heldout"
        examples.append(make_example(candidate, directory, split))
    train = [r for r in examples if r["metadata"]["split"] == "train"]
    heldout = [r for r in examples if r["metadata"]["split"] == "heldout"]
    if len(train) != 13 or len(heldout) != 8:
        raise ValueError(f"Unexpected split size: {len(train)}/{len(heldout)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("heldout", heldout)):
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "source_url": examples[0]["metadata"]["source_url"],
        "source_pdf_sha256": source_pdf_hashes.pop(),
        "license": "CC0 1.0 Universal",
        "train_cores": sorted(TRAIN_CORES), "heldout_cores": sorted(HELDOUT_CORES),
        "train_windows": len(train), "heldout_windows": len(heldout),
        "training_approved": 0,
        "split_rule": "Entire core IDs disjoint between train and heldout",
        "scope": "Experimental visual-structure description of Fire Island sand; not lithology diversity or facies accuracy",
        "excluded_after_visual_review": ["P10-C03 30-37 cm", "P29-C11 143-148 cm"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
