"""Write a reproducible model card for the isolated Cat Island pilot adapter."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cat_island_lithology_pilot"
ADAPTER = ROOT / "outputs" / "core_vlm_cat_island_lithology_pilot_lora"
EVALUATION = ROOT / "outputs" / "cat_island_lithology_pilot_evaluation.json"
OUTPUT = ROOT / "model_spec_cat_island_lithology_pilot.json"
FORCED_DATA = ROOT / "data" / "cat_island_four_class_pilot"
FORCED_ADAPTER = ROOT / "outputs" / "core_vlm_cat_island_four_class_pilot_lora"
FORCED_EVALUATION = ROOT / "outputs" / "cat_island_four_class_evaluation.json"
FORCED_OUTPUT = ROOT / "model_spec_cat_island_four_class_pilot.json"


def main() -> None:
    selection = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    training = json.loads((ADAPTER / "training_metrics.json").read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))["summary"]
    card = {
        "name": "Cat Island Russian lithology pilot",
        "date": "2026-10-11",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "adapter_path": str(ADAPTER),
        "source": "USGS Data Series 834 (10.3133/ds834)",
        "source_url": "https://pubs.usgs.gov/ds/0834/ds834_data.html",
        "task": "Russian JSON lithology of a cropped sediment-core interval",
        "scope": "sand, mud, clay, peat and related unconsolidated coastal sediments; not a full geological core description or depositional-setting model",
        "training": {"train_images": selection["train_images"],
                     "train_cores": selection["train_cores"],
                     "epochs": 2, "learning_rate": 5e-6,
                     "train_loss": training["train_loss"],
                     "training_approved_rows_used": training["training_approved_rows_used"]},
        "holdout": evaluation,
        "review_status": "experimental_unreviewed_pdf_extraction_and_crop",
        "deployment_status": "not_in_core_sedimentologist_app",
        "limitations": [
            "USGS descriptions were automatically parsed and depth-cropped, then lithology names were translated by a fixed dictionary.",
            "The source labels and crops require geological review; some lithologies are visually ambiguous.",
            "Exact-class accuracy on seven held-out cores is not accuracy of full descriptions or depositional environments.",
            "The E: local drive was unavailable during this run; user core photos were not added to this pilot.",
        ],
    }
    OUTPUT.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    forced_selection = json.loads((FORCED_DATA / "summary.json").read_text(encoding="utf-8"))
    forced_training = json.loads((FORCED_ADAPTER / "training_metrics.json").read_text(encoding="utf-8"))
    forced_evaluation = json.loads(FORCED_EVALUATION.read_text(encoding="utf-8"))["summary"]
    forced_card = {
        "name": "Cat Island forced four-class lithology diagnostic",
        "date": "2026-10-11",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "adapter_path": str(FORCED_ADAPTER),
        "source": "USGS Data Series 834 (10.3133/ds834)",
        "source_url": "https://pubs.usgs.gov/ds/0834/ds834_data.html",
        "task": "Forced choice between four Russian lithology labels on an interval crop",
        "classes": forced_selection["classes"],
        "training": {"train_images": forced_selection["counts"]["train"],
                     "epochs": 2, "learning_rate": 5e-6,
                     "train_loss": forced_training["train_loss"],
                     "training_approved_rows_used": forced_training["training_approved_rows_used"]},
        "holdout": forced_evaluation,
        "review_status": "experimental_unreviewed_pdf_extraction_and_crop",
        "deployment_status": "rejected_for_core_sedimentologist_app",
        "reason": "Only 2/27 exact classes on held-out cores; adapter gives one class for every photo and does not beat base Qwen.",
    }
    FORCED_OUTPUT.write_text(json.dumps(forced_card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(FORCED_OUTPUT)


if __name__ == "__main__":
    main()
