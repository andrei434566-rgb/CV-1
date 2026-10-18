"""Make a forced-choice diagnostic from the Cat Island lithology candidates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "cat_island_lithology_pilot"
OUT = ROOT / "data" / "cat_island_four_class_pilot"
CLASSES = {"песок", "глина", "глинистый песок", "ил"}
PROMPT = (
    "<image>\nОпредели наиболее вероятную преобладающую породу на фрагменте фото керна. "
    "Выбери ровно один вариант: песок, глина, глинистый песок, ил. "
    "Это исследовательская классификация; даже при сомнении выбери наиболее вероятный вариант. "
    "Ответь только JSON с ключом lithology на русском языке."
)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    counts = {}
    for split in ("train", "heldout"):
        original = [json.loads(line) for line in (SOURCE / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        rows = []
        for row in original:
            if row["metadata"]["lithology_target_ru"] not in CLASSES:
                continue
            row["conversations"][0]["value"] = PROMPT
            row["metadata"]["experiment"] = "forced_four_class_diagnostic"
            rows.append(row)
        (OUT / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
        )
        counts[split] = len(rows)
    (OUT / "summary.json").write_text(json.dumps({"classes": sorted(CLASSES), "counts": counts,
                                                   "purpose": "diagnostic only; forced answers are unsuitable for unsupervised geological use",
                                                   "training_approved": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
