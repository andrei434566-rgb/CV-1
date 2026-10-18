"""Compare fixed-well 8k and 40k crop classifiers on identical records."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "outputs/core_cv_mendeley_resnet18_imagenet_8000"
NEW = ROOT / "outputs/core_cv_mendeley_resnet18_imagenet_40000"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def two_sided_exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if not discordant:
        return 1.0
    probability = 2 * sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / 2**discordant
    return min(1.0, probability)


def compare(split: str) -> dict:
    old = load(OLD / f"{split}_evaluation.json")
    new = load(NEW / f"{split}_evaluation.json")
    old_rows = {row["record_id"]: row for row in old["records"]}
    new_rows = {row["record_id"]: row for row in new["records"]}
    if len(old_rows) != 400 or old_rows.keys() != new_rows.keys():
        raise ValueError("Different evaluation records")
    old_only = new_only = both_correct = both_wrong = 0
    switches = []
    for record_id, before in old_rows.items():
        after = new_rows[record_id]
        if (before["well_id"], before["expected_en"], before["source_archive_member"]) != (
                after["well_id"], after["expected_en"], after["source_archive_member"]):
            raise ValueError(f"Mismatched label or well: {record_id}")
        old_ok = before["predicted_en"] == before["expected_en"]
        new_ok = after["predicted_en"] == after["expected_en"]
        if old_ok and new_ok:
            both_correct += 1
        elif old_ok:
            old_only += 1
        elif new_ok:
            new_only += 1
        else:
            both_wrong += 1
        if old_ok != new_ok:
            switches.append({"record_id": record_id, "well_id": before["well_id"],
                             "expected": before["expected_en"],
                             "old_prediction": before["predicted_en"],
                             "new_prediction": after["predicted_en"]})
    return {
        "split": split,
        "old_train_count": 8000,
        "new_train_count": 40000,
        "old_score": old["score"],
        "new_score": new["score"],
        "paired_correct": {"both": both_correct, "old_only": old_only,
                           "new_only": new_only, "neither": both_wrong},
        "mcnemar_exact_two_sided_p": two_sided_exact_mcnemar(old_only, new_only),
        "per_well": {well: {"old_correct": old["per_well"][well]["correct"],
                            "new_correct": new["per_well"][well]["correct"],
                            "total": old["per_well"][well]["count"]}
                     for well in sorted(old["per_well"])},
        "changed_outcome_records": switches,
    }


def main() -> None:
    result = {
        "scope": "four-class 2x2 cm Mendeley crops only; not geological description/facies",
        "validation": compare("val"),
        "test": compare("test"),
        "primary_model_decision": "retain_8k_until_new_independent_well_validation",
        "reason": "40k validation is one crop lower; 40k test is four crops higher on a small, previously explored well set; effect is not robust enough for application promotion",
    }
    (NEW / "comparison_with_8000.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        split: {"old_correct": result[split]["old_score"]["correct"],
                "new_correct": result[split]["new_score"]["correct"],
                "paired": result[split]["paired_correct"],
                "mcnemar_p": result[split]["mcnemar_exact_two_sided_p"]}
        for split in ("validation", "test")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
