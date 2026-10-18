"""Summarize validation of the expanded Mendeley lithology continuation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "outputs/mendeley_lithology_expanded_val_evaluation.json"
PILOT_SAME_VAL = ROOT / "outputs/mendeley_lithology_pilot_on_expanded_val_evaluation.json"
ADAPTER = ROOT / "outputs/core_vlm_mendeley_lithology_expanded_lora/adapter_model.safetensors"
DATA = ROOT / "data/mendeley_lithology_expanded"
SOURCE = ROOT / "data/external_raw/mendeley_lithology/inventory.json"


def read_rows(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metrics = json.loads((ADAPTER.parent / "training_metrics.json").read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    train, val = read_rows("train"), read_rows("val")
    assert report["split"] == "val" and len(train) == 2000 and len(val) == 200
    assert report["score"]["count"] == len(val)
    assert not set(report["train_wells"]) & set(report["heldout_wells"])
    classes = defaultdict(set)
    for row in train:
        classes[row["metadata"]["source_label_en"]].add(row["metadata"]["well_id"])

    errors = [row for row in report["records"] if row["adapter_label"] != row["expected"]]
    queue = ROOT / "outputs/mendeley_lithology_expanded_val_errors.csv"
    with queue.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "record_id", "well_id", "image", "source_archive_member", "source_label_ru",
            "base_prediction", "adapter_prediction", "review_status", "geologist_corrected_label", "reviewer_name", "reviewer_note",
        ])
        writer.writeheader()
        for row in errors:
            writer.writerow({
                "record_id": row["record_id"], "well_id": row["well_id"],
                "image": row["image"], "source_archive_member": row["source_archive_member"],
                "source_label_ru": row["expected"], "base_prediction": row["base_label"] or row["base_raw"],
                "adapter_prediction": row["adapter_label"] or row["adapter_raw"],
                "review_status": "pending", "geologist_corrected_label": "",
                "reviewer_name": "", "reviewer_note": "",
            })
    card = {
        "name": "core-vlm-mendeley-lithology-expanded",
        "date": "2026-10-12",
        "status": "experimental_validation_only_not_integrated",
        "task": "Russian four-class lithology/non-core classification on 2x2 cm crops, not facies or description",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "initial_adapter": "outputs/core_vlm_mendeley_lithology_pilot_lora/adapter_model.safetensors",
        "adapter_path": str(ADAPTER.relative_to(ROOT)).replace("\\", "/"),
        "adapter_sha256": hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
        "source": {"url": source["source_url"], "doi": source["doi"], "license": source["license"], "archive_sha256": source["archive_sha256"]},
        "training": {
            "manifest": "data/mendeley_lithology_expanded/train.jsonl",
            "train_crops": metrics["candidate_rows_used"],
            "train_wells": metrics["independent_wells"],
            "train_wells_by_class": {label: len(classes[label]) for label in ("sandstone", "limestone", "shale", "garbage")},
            "geologist_approved_rows": metrics["geology_reviewed_rows_used"],
            "epochs": metrics["epoch"], "learning_rate": 0.000005,
            "train_loss": metrics["train_loss"],
            "training_status": metrics["training_status"],
        },
        "validation": {
            "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "wells": report["heldout_wells"], "crops": report["score"]["count"],
            "base_correct": report["score"]["base_correct"],
            "adapter_correct": report["score"]["adapter_correct"],
            "base_macro_f1": report["score"]["base_macro_f1"],
            "adapter_macro_f1": report["score"]["adapter_macro_f1"],
            "adapter_recall_by_class": report["score"]["adapter_recall_by_class"],
            "per_well": {well: {"correct": row["adapter_correct"], "total": row["count"]} for well, row in report["per_well"].items()},
            "error_review_csv": str(queue.relative_to(ROOT)).replace("\\", "/"),
            "decision": "Validation is for development only; no claim of independent external accuracy or full geological-description capability.",
        },
        "limitations": [
            "Validation wells can guide development and are not a sealed final test.",
            "The four formerly sealed test wells were already used for the previous pilot and were not used for this training or validation run.",
            "Source-folder labels are not individually geologist reviewed.",
            "Five independent training wells supply limestone crops; adjacent crops are correlated.",
            "No facies, depositional environment or Russian free-text description labels exist in this dataset.",
            "Ancient-core transfer and geologist review are required before integration.",
        ],
    }
    if PILOT_SAME_VAL.is_file():
        pilot = json.loads(PILOT_SAME_VAL.read_text(encoding="utf-8"))
        assert [(r["record_id"], r["expected"]) for r in pilot["records"]] == [(r["record_id"], r["expected"]) for r in report["records"]]
        card["validation"]["pilot_adapter_correct_on_same_images"] = pilot["score"]["adapter_correct"]
        card["validation"]["pilot_adapter_macro_f1_on_same_images"] = pilot["score"]["adapter_macro_f1"]
        card["validation"]["pilot_comparison_report"] = str(PILOT_SAME_VAL.relative_to(ROOT)).replace("\\", "/")
    output = ROOT / "model_spec_mendeley_lithology_expanded.json"
    output.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"val_correct": card["validation"]["adapter_correct"], "val_crops": card["validation"]["crops"], "errors": len(errors)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
