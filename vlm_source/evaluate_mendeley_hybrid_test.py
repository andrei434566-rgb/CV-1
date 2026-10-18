"""Apply the frozen validation-defined CV/VLM decision rule once to new test crops."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FROZEN_VAL = ROOT / "outputs/mendeley_cv_vlm_validation_comparison.json"
CNN = ROOT / "outputs/core_cv_mendeley_resnet18_8000/test_evaluation.json"
VLM = ROOT / "outputs/mendeley_lithology_8000_new_test.json"
OUTPUT = ROOT / "outputs/mendeley_cv_vlm_new_test_evaluation.json"
EN_TO_RU = {"sandstone": "песчаник", "limestone": "известняк",
            "shale": "глинистая порода", "garbage": "не керн"}
EXPECTED_RULE = "Use CNN if it predicts shale or garbage, otherwise use Qwen VLM class"


def score(rows: list[dict], key: str) -> dict:
    classes = {}
    f1s = []
    for label in EN_TO_RU.values():
        tp = sum(r["expected"] == label and r[key] == label for r in rows)
        fp = sum(r["expected"] != label and r[key] == label for r in rows)
        fn = sum(r["expected"] == label and r[key] != label for r in rows)
        classes[label] = {"correct": tp, "total": tp + fn,
                          "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0}
        f1s.append(classes[label]["f1"])
    return {"correct": sum(r[key] == r["expected"] for r in rows), "total": len(rows),
            "macro_f1": sum(f1s) / len(f1s), "per_class": classes}


def main() -> None:
    frozen = json.loads(FROZEN_VAL.read_text(encoding="utf-8"))
    if frozen["hybrid_rule"] != EXPECTED_RULE or frozen["hybrid"]["total"] != 400:
        raise ValueError("Rule was not fixed on the prior validation set")
    cnn = json.loads(CNN.read_text(encoding="utf-8"))
    vlm = json.loads(VLM.read_text(encoding="utf-8"))
    if cnn["split"] != "test" or vlm["split"] != "test" or len(cnn["records"]) != 400:
        raise ValueError("Expected one 400-crop test split")
    by_id = {row["record_id"]: row for row in cnn["records"]}
    if len(by_id) != len(vlm["records"]):
        raise ValueError("Test row count mismatch")
    records = []
    for row in vlm["records"]:
        other = by_id[row["record_id"]]
        expected = EN_TO_RU[other["expected_en"]]
        if (row["expected"] != expected or row["well_id"] != other["well_id"]
                or row["source_archive_member"] != other["source_archive_member"]):
            raise ValueError("Mismatched image or source target")
        cnn_label = EN_TO_RU[other["predicted_en"]]
        hybrid = cnn_label if other["predicted_en"] in {"shale", "garbage"} else row["adapter_label"]
        records.append({"record_id": row["record_id"], "well_id": row["well_id"],
                        "image": row["image"], "source_archive_member": row["source_archive_member"],
                        "expected": expected, "vlm": row["adapter_label"], "cnn": cnn_label,
                        "hybrid": hybrid})
    wells = defaultdict(list)
    for row in records:
        wells[row["well_id"]].append(row)
    outcomes = Counter((r["vlm"] == r["expected"], r["cnn"] == r["expected"]) for r in records)
    report = {
        "source": "https://data.mendeley.com/datasets/b6rj29mrss/1",
        "scope": "Four-class lithology/non-core on 2x2 cm crops, not description or facies",
        "frozen_rule_before_test": EXPECTED_RULE,
        "development_rule_source": str(FROZEN_VAL.relative_to(ROOT)).replace("\\", "/"),
        "test_images": len(records), "test_wells": sorted(wells),
        "vlm": score(records, "vlm"), "cnn": score(records, "cnn"),
        "hybrid": score(records, "hybrid"),
        "per_well": {well: {mode: score(rows, mode) for mode in ("vlm", "cnn", "hybrid")}
                     for well, rows in sorted(wells.items())},
        "outcomes": {"both_correct": outcomes[(True, True)],
                     "vlm_only_correct": outcomes[(True, False)],
                     "cnn_only_correct": outcomes[(False, True)],
                     "both_wrong": outcomes[(False, False)]},
        "limitations": [
            "These 400 image files were unused, but their four well identities appeared in an earlier pilot; this is not a completely new-well deployment test.",
            "Source folder labels have not been individually reviewed by a geologist.",
            "The rule was chosen on the separate 400-crop development validation before this test; no test-based retuning is allowed.",
            "The models only output the narrow source lithology class; no full description, texture, contacts, facies or depositional environment.",
        ],
        "records": records,
    }
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving original test result: {OUTPUT}")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mode: report[mode] for mode in ("vlm", "cnn", "hybrid")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
