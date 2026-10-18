"""Whole-well evaluation for the experimental five-class UKCS Qwen adapter."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_barnegat_binary_pilot import MODEL, ROOT, predict
from inspect_ukcs_labels import LABELS


DATA = ROOT / "data" / "ukcs_q204_q205_lithology_candidates"
ADAPTER = ROOT / "outputs" / "core_vlm_ukcs_lithology_pilot_lora"
OUTPUT = ROOT / "outputs" / "ukcs_lithology_pilot_evaluation.json"
CLASSES = tuple(LABELS.values())


def read_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(answer: str) -> str | None:
    cleaned = answer.lower().strip().strip(' .,:;!"«»')
    return cleaned if cleaned in CLASSES else None


def score(rows: list[dict], majority: str) -> dict:
    result = {
        "count": len(rows),
        "wells": len({r["well_id"] for r in rows}),
        "expected_distribution": dict(Counter(r["expected"] for r in rows)),
        "majority_class": majority,
        "majority_correct": sum(r["expected"] == majority for r in rows),
        "base_correct": sum(r["base_label"] == r["expected"] for r in rows),
        "adapter_correct": sum(r["adapter_label"] == r["expected"] for r in rows),
    }
    for mode in ("base", "adapter"):
        f1 = []
        recall = {}
        for label in CLASSES:
            tp = sum(r["expected"] == label and r[f"{mode}_label"] == label for r in rows)
            fp = sum(r["expected"] != label and r[f"{mode}_label"] == label for r in rows)
            fn = sum(r["expected"] == label and r[f"{mode}_label"] != label for r in rows)
            f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
            recall[label] = tp / (tp + fn) if tp + fn else None
        result[f"{mode}_macro_f1"] = sum(f1) / len(f1)
        result[f"{mode}_recall_by_class"] = recall
        result[f"{mode}_prediction_distribution"] = dict(Counter(r[f"{mode}_label"] or "неопределено" for r in rows))
    return result


def collapse_runs(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_run_id"]].append(row)
    collapsed = []
    for run_id, tiles in groups.items():
        item = {"source_run_id": run_id, "well_id": tiles[0]["well_id"], "expected": tiles[0]["expected"], "tiles": len(tiles)}
        for mode in ("base", "adapter"):
            counts = Counter(tile[f"{mode}_label"] for tile in tiles)
            best = counts.most_common()
            item[f"{mode}_label"] = best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None
        collapsed.append(item)
    return collapsed


def main() -> None:
    train = read_rows("train_pilot.jsonl")
    holdout = read_rows("holdout_pilot.jsonl")
    train_wells = {r["metadata"]["well_id"] for r in train}
    holdout_wells = {r["metadata"]["well_id"] for r in holdout}
    if train_wells & holdout_wells:
        raise ValueError("Train and holdout share wells")
    majority = Counter(r["conversations"][1]["value"] for r in train).most_common(1)[0][0]
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    base.eval()
    results = []
    for index, row in enumerate(holdout, 1):
        answer = predict(base, processor, row)
        meta = row["metadata"]
        results.append({
            "record_id": meta["record_id"],
            "well_id": meta["well_id"],
            "source_run_id": meta["source_run_id"],
            "depth_top_m": meta["depth_top_m"],
            "depth_bottom_m": meta["depth_bottom_m"],
            "image": row["image"],
            "expected": row["conversations"][1]["value"],
            "base_raw": answer,
            "base_label": normalize(answer),
        })
        print(f"base {index}/{len(holdout)}", flush=True)
    adapter = PeftModel.from_pretrained(base, ADAPTER)
    adapter.eval()
    for index, (row, result) in enumerate(zip(holdout, results), 1):
        answer = predict(adapter, processor, row)
        result["adapter_raw"] = answer
        result["adapter_label"] = normalize(answer)
        print(f"adapter {index}/{len(holdout)}", flush=True)
    runs = collapse_runs(results)
    report = {
        "scope": "Five-class UKCS lithology from photos only; no depositional environment or full text description.",
        "train_wells": sorted(train_wells),
        "holdout_wells": sorted(holdout_wells),
        "train_images": len(train),
        "tile_level": score(results, majority),
        "original_run_level": score(runs, majority),
        "per_well": {well: score([r for r in results if r["well_id"] == well], majority) for well in sorted(holdout_wells)},
        "limitations": [
            "Candidates and Russian translations were not approved by a geologist.",
            "Two crops may come from the same contiguous lithology run; report includes both tile and run units.",
            "Split by entire well, but all wells are from a related deep-marine province; external transfer still required.",
        ],
        "run_records": runs,
        "records": results,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("tile_level", "original_run_level")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
