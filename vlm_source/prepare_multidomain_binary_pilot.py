"""Combine two public core-photo domains for a provisional binary VLM pilot."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluate_ukcs_binary_transfer import MAP


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "multidomain_binary_pilot"
PROMPT = (
    "По этому фрагменту фотографии керна выбери преобладающий материал. "
    "Ответь ровно одним словом из списка: песчаный, тонкозернистый."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert(row: dict, source: str) -> dict:
    result = json.loads(json.dumps(row, ensure_ascii=False))
    original = result["conversations"][1]["value"]
    expected = MAP[original] if source == "ukcs" else original
    if expected not in ("песчаный", "тонкозернистый"):
        raise ValueError((source, expected))
    result["conversations"][0]["value"] = "<image>\n" + PROMPT
    result["conversations"][1]["value"] = expected
    meta = result["metadata"]
    meta["binary_source"] = source
    meta["original_source_label"] = original
    meta["training_approved"] = False
    return result


def main() -> None:
    ukcs = ROOT / "data" / "ukcs_q204_q205_lithology_candidates"
    barnegat = ROOT / "data" / "barnegat_binary_pilot"
    train = [convert(row, "ukcs") for row in read_jsonl(ukcs / "train_pilot.jsonl")]
    train += [convert(row, "barnegat") for row in read_jsonl(barnegat / "train_candidates.jsonl")]
    holdout = [convert(row, "ukcs") for row in read_jsonl(ukcs / "holdout_pilot.jsonl")]
    holdout += [convert(row, "barnegat") for row in read_jsonl(barnegat / "holdout_candidates.jsonl")]
    train_groups = {(r["metadata"]["binary_source"], r["metadata"]["well_id"]) for r in train}
    holdout_groups = {(r["metadata"]["binary_source"], r["metadata"]["well_id"]) for r in holdout}
    if train_groups & holdout_groups:
        raise ValueError("A source well appears in both train and holdout")
    OUTPUT.mkdir(exist_ok=True)
    for name, rows in (("train_candidates.jsonl", train), ("holdout_candidates.jsonl", holdout)):
        (OUTPUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {
        "train_crops": len(train),
        "train_source_wells": len(train_groups),
        "train_classes": dict(Counter(r["conversations"][1]["value"] for r in train)),
        "train_sources": dict(Counter(r["metadata"]["binary_source"] for r in train)),
        "holdout_crops": len(holdout),
        "holdout_source_wells": len(holdout_groups),
        "holdout_classes": dict(Counter(r["conversations"][1]["value"] for r in holdout)),
        "holdout_sources": dict(Counter(r["metadata"]["binary_source"] for r in holdout)),
        "source_well_overlap": 0,
        "external_unseen_datasets": ["r41_ancient_core", "cat_island_modern_core"],
        "geologist_approved": 0,
        "status": "experimental candidates only; no geologist approval",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
