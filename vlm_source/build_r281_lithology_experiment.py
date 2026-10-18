"""Build a core-disjoint R-281 cut-face lithology classification experiment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "r281_slices" / "records.jsonl"
OUT = ROOT / "data" / "r281_lithology_experiment"
TEST_CORES = {2, 9}
CLASSES = {"песчаник", "алевролит", "аргиллит", "уголь", "переслаивание"}
PROMPT = (
    "<image>\nПеред тобой фото торца одного метрового среза керна. "
    "Определи основную литологию только по видимому изображению. "
    "Выбери одно из: песчаник, алевролит, аргиллит, уголь, переслаивание. "
    "Ответь на русском строго JSON вида {\"rock_type\":\"название\"}. "
    "Если признаки неразличимы, ответь {\"rock_type\":\"неопределено\"}. "
    "Не используй фон, наклейки и цветное освещение для определения породы."
)


def write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    train, test = [], []
    for row in source:
        rock = row["rock_first_word"]
        if rock not in CLASSES:
            raise ValueError(f"Unexpected lithology: {rock}")
        record = {
            "image": row["image"],
            "conversations": [
                {"from": "human", "value": PROMPT},
                {"from": "gpt", "value": json.dumps({"rock_type": rock}, ensure_ascii=False)},
            ],
            "metadata": {
                "record_id": row["record_id"], "record_type": "r281_embedded_cut_face_candidate",
                "source_group": "user_drive_R281_DOCX", "well_id": row["well_id"],
                "core_number": row["core_number"], "cut_number": row["cut_number"],
                "depth_m": row["depth_m"], "rock_type": rock,
                "source_description": row["description"],
                "source_document_member": row["source_document_member"],
                "source_image_member": row["source_image_member"],
                "training_approved": False, "geology_approved": False,
                "label_rule": "first rock name or explicit interbedding in adjacent report sentence",
                "warning": "Text is an adjacent report label; not all lithology or composition is visible in this small blue-tinted photo.",
            },
        }
        (test if row["core_number"] in TEST_CORES else train).append(record)
    if {r["metadata"]["core_number"] for r in train} & {r["metadata"]["core_number"] for r in test}:
        raise ValueError("Core leakage")
    write(OUT / "train.jsonl", train)
    write(OUT / "test.jsonl", test)
    summary = {
        "train_records": len(train), "test_records": len(test),
        "train_cores": sorted({r["metadata"]["core_number"] for r in train}),
        "test_cores": sorted(TEST_CORES),
        "train_class_counts": dict(Counter(r["metadata"]["rock_type"] for r in train)),
        "test_class_counts": dict(Counter(r["metadata"]["rock_type"] for r in test)),
        "majority_baseline_test_accuracy": max(Counter(r["metadata"]["rock_type"] for r in test).values()) / len(test),
        "scope": "Diagnostic cut-face lithology classification within one R-281 well; not facies or full description.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
