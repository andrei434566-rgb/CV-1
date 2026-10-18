"""Inspect complementary validation errors and one fixed class-specialist rule."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VLM = ROOT / "outputs/mendeley_lithology_next_posttrain_val.json"
CNN = ROOT / "outputs/core_cv_mendeley_resnet18_8000/val_evaluation.json"
OUTPUT = ROOT / "outputs/mendeley_cv_vlm_validation_comparison.json"
EN_TO_RU = {"sandstone": "песчаник", "limestone": "известняк",
            "shale": "глинистая порода", "garbage": "не керн"}


def metric(rows: list[dict], key: str) -> dict:
    score = {"correct": sum(row[key] == row["expected"] for row in rows), "total": len(rows)}
    recalls = {}
    for name in EN_TO_RU.values():
        subset = [row for row in rows if row["expected"] == name]
        recalls[name] = {"correct": sum(row[key] == name for row in subset), "total": len(subset)}
    score["per_class"] = recalls
    return score


def main() -> None:
    vlm = json.loads(VLM.read_text(encoding="utf-8"))
    cnn = json.loads(CNN.read_text(encoding="utf-8"))
    by_id = {row["record_id"]: row for row in cnn["records"]}
    if len(by_id) != len(cnn["records"]) or len(by_id) != len(vlm["records"]):
        raise ValueError("Validation record mismatch")
    joined = []
    for row in vlm["records"]:
        other = by_id[row["record_id"]]
        expected = EN_TO_RU[other["expected_en"]]
        if (row["expected"] != expected or row["well_id"] != other["well_id"]
                or row["source_archive_member"] != other["source_archive_member"]):
            raise ValueError("Unequal source row or target")
        cnn_label = EN_TO_RU[other["predicted_en"]]
        vlm_label = row["adapter_label"]
        # Fixed hypothesis: the CNN is highly precise for shale/non-core;
        # otherwise retain the VLM, which has stronger limestone recall.
        hybrid = cnn_label if other["predicted_en"] in {"shale", "garbage"} else vlm_label
        joined.append({"record_id": row["record_id"], "well_id": row["well_id"],
                       "expected": expected, "vlm": vlm_label, "cnn": cnn_label,
                       "cnn_softmax_max_uncalibrated": other["model_softmax_max_uncalibrated"],
                       "hybrid": hybrid})
    outcomes = Counter()
    for row in joined:
        outcomes[(row["vlm"] == row["expected"], row["cnn"] == row["expected"])] += 1
    report = {
        "scope": "Development validation only; rule is a research hypothesis, not a final deployed model",
        "hybrid_rule": "Use CNN if it predicts shale or garbage, otherwise use Qwen VLM class",
        "vlm": metric(joined, "vlm"), "cnn": metric(joined, "cnn"),
        "hybrid": metric(joined, "hybrid"),
        "agreement_count": sum(row["vlm"] == row["cnn"] for row in joined),
        "outcomes": {"both_correct": outcomes[(True, True)],
                     "vlm_only_correct": outcomes[(True, False)],
                     "cnn_only_correct": outcomes[(False, True)],
                     "both_wrong": outcomes[(False, False)]},
        "records": joined,
    }
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving prior comparison: {OUTPUT}")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("vlm", "cnn", "hybrid", "agreement_count", "outcomes")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
