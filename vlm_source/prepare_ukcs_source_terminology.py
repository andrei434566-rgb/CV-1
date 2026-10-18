"""Use the paper's five source classes with conservative Russian names."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NEW = ROOT / "data/ukcs_q204_q205_lithology_expanded"
OLD = ROOT / "data/ukcs_q204_q205_lithology_candidates"
NAMES = {
    "s": ("sandstone", "песчаник"),
    "is": ("muddy sandstone", "глинистый песчаник"),
    "ih": ("sandy mudstone", "песчанистая глинистая порода"),
    "os": ("oil-stained sandstone", "песчаник с нефтяными пятнами"),
    "sh": ("mudstone", "глинистая порода"),
}
PROMPT = (
    "<image>\nОпредели один исходный литологический класс на фрагменте фото керна. "
    "Варианты: песчаник, глинистый песчаник, песчанистая глинистая порода, "
    "песчаник с нефтяными пятнами, глинистая порода. "
    "Ответь только точным названием одного класса на русском языке."
)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def convert(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        meta = row["metadata"].copy()
        code = meta["source_label_code"]
        english, russian = NAMES[code]
        meta.update({
            "source_label_english_paper": english,
            "translated_label_ru": russian,
            "translation_status": "editorial_conservative_geologist_review_pending",
            "training_approved": False,
        })
        converted.append({
            "image": row["image"],
            "conversations": [
                {"from": "human", "value": PROMPT},
                {"from": "gpt", "value": russian},
            ],
            "metadata": meta,
        })
    return converted


def main() -> None:
    train = convert(read(NEW / "train_pilot.jsonl"))
    fixed = convert(read(OLD / "holdout_pilot.jsonl"))
    train_wells = {r["metadata"]["well_id"] for r in train}
    fixed_wells = {r["metadata"]["well_id"] for r in fixed}
    if train_wells & fixed_wells or len(train_wells) != 9 or len(fixed_wells) != 3:
        raise ValueError("Whole-well separation failed")
    for path, rows in ((NEW / "train_source_terminology.jsonl", train),
                       (NEW / "holdout_fixed_source_terminology.jsonl", fixed)):
        if path.exists():
            raise FileExistsError(f"Preserving existing labels: {path}")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "paper": "https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2021.659611/full",
        "source_archive": "https://figshare.com/articles/dataset/UKCS_Q204_Q205_Subsurface_Data_products_for_Machine_Learning_Study/14265689",
        "class_mapping": {code: {"english_paper": en, "russian": ru} for code, (en, ru) in NAMES.items()},
        "train_crops": len(train), "train_wells": sorted(train_wells),
        "fixed_holdout_crops": len(fixed), "fixed_holdout_wells": sorted(fixed_wells),
        "geologist_approved": 0,
        "status": "experimental source-aligned class training; not full descriptions or depositional facies",
    }
    (NEW / "source_terminology.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "fixed_holdout": len(fixed), "class_mapping": report["class_mapping"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
