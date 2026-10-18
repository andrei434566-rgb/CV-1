"""Compare EfficientNet-B0 against the retained ResNet-18 on identical wells."""

from __future__ import annotations

import json
from pathlib import Path

from compare_mendeley_40000 import two_sided_exact_mcnemar


ROOT = Path(__file__).resolve().parent / "outputs"
OLD = ROOT / "core_cv_mendeley_resnet18_imagenet_8000"
NEW = ROOT / "core_cv_mendeley_efficientnet_b0_40000"


def compare(split: str) -> dict:
    old = json.loads((OLD / f"{split}_evaluation.json").read_text(encoding="utf-8"))
    new = json.loads((NEW / f"{split}_evaluation.json").read_text(encoding="utf-8"))
    before = {row["record_id"]: row for row in old["records"]}
    after = {row["record_id"]: row for row in new["records"]}
    if len(before) != 400 or before.keys() != after.keys():
        raise ValueError("Evaluation records differ")
    both = old_only = new_only = neither = 0
    changed = []
    for record_id, row in before.items():
        candidate = after[record_id]
        identity = ("well_id", "expected_en", "source_archive_member")
        if any(row[key] != candidate[key] for key in identity):
            raise ValueError(f"Record identity or source label mismatch: {record_id}")
        old_ok = row["expected_en"] == row["predicted_en"]
        new_ok = candidate["expected_en"] == candidate["predicted_en"]
        if old_ok and new_ok:
            both += 1
        elif old_ok:
            old_only += 1
        elif new_ok:
            new_only += 1
        else:
            neither += 1
        if old_ok != new_ok:
            changed.append({"record_id": record_id, "well_id": row["well_id"],
                            "expected_en": row["expected_en"],
                            "old_predicted_en": row["predicted_en"],
                            "new_predicted_en": candidate["predicted_en"]})
    return {"split": split, "old_score": old["score"], "new_score": new["score"],
            "paired_correct": {"both": both, "old_only": old_only,
                               "new_only": new_only, "neither": neither},
            "mcnemar_exact_two_sided_p": two_sided_exact_mcnemar(old_only, new_only),
            "per_well": {well: {"old_correct": old["per_well"][well]["correct"],
                                "new_correct": new["per_well"][well]["correct"],
                                "total": old["per_well"][well]["count"]}
                         for well in sorted(old["per_well"])},
            "changed_outcome_records": changed}


def main() -> None:
    result = {
        "scope": "four-class, 2x2-cm Mendeley crops only; no description or facies evaluation",
        "baseline": "ResNet-18 ImageNet, 8000 train crops",
        "candidate": "EfficientNet-B0 ImageNet, 40000 train crops",
        "validation": compare("val"),
        "test": compare("test"),
        "limitations": ["Only four heldout wells per split; crops from a well are correlated.",
                        "These heldout well groups appeared in earlier project experiments.",
                        "The source labels have not been individually reviewed by a geologist."],
    }
    output = NEW / "comparison_with_8000.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: {"old_correct": result[split]["old_score"]["correct"],
                              "new_correct": result[split]["new_score"]["correct"],
                              "paired": result[split]["paired_correct"],
                              "mcnemar_p": result[split]["mcnemar_exact_two_sided_p"]}
                      for split in ("validation", "test")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
