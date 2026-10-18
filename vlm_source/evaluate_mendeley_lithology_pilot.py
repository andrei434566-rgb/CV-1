"""Independent whole-well test of the source-labeled Mendeley lithology pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_barnegat_binary_pilot import MODEL, ROOT, predict


DATA = ROOT / "data/mendeley_lithology_pilot"
ADAPTER = ROOT / "outputs/core_vlm_mendeley_lithology_pilot_lora"
LABELS = ("песчаник", "известняк", "глинистая порода", "не керн")


def read_rows(data_dir, split: str) -> list[dict]:
    return [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]


def normalize(raw: str) -> str | None:
    value = raw.lower().strip().strip(' .,:;!"«»')
    return value if value in LABELS else None


def score(rows: list[dict]) -> dict:
    have_base = all("base_label" in row for row in rows)
    result = {
        "count": len(rows),
        "wells": len({row["well_id"] for row in rows}),
        "expected_distribution": dict(Counter(row["expected"] for row in rows)),
        "majority_correct": max(Counter(row["expected"] for row in rows).values()),
        "adapter_correct": sum(row["adapter_label"] == row["expected"] for row in rows),
    }
    if have_base:
        result["base_correct"] = sum(row["base_label"] == row["expected"] for row in rows)
    for model in (("base", "adapter") if have_base else ("adapter",)):
        f1s = []
        recall = {}
        for label in LABELS:
            tp = sum(row["expected"] == label and row[f"{model}_label"] == label for row in rows)
            fp = sum(row["expected"] != label and row[f"{model}_label"] == label for row in rows)
            fn = sum(row["expected"] == label and row[f"{model}_label"] != label for row in rows)
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
            recall[label] = tp / (tp + fn) if tp + fn else None
        result[f"{model}_macro_f1"] = sum(f1s) / len(f1s)
        result[f"{model}_recall_by_class"] = recall
        result[f"{model}_prediction_distribution"] = dict(Counter(row[f"{model}_label"] or "неопределено" for row in rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--train-manifest", type=Path, help="Training JSONL when validation resides in another split directory")
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--reuse-base-report", type=Path, help="Prior report on these exact heldout rows; reuse only its base predictions")
    parser.add_argument("--skip-base", action="store_true", help="Evaluate the adapter only; no base prediction or score")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.skip_base and args.reuse_base_report:
        parser.error("--skip-base and --reuse-base-report cannot be combined")
    train = ([json.loads(line) for line in args.train_manifest.read_text(encoding="utf-8").splitlines() if line]
             if args.train_manifest else read_rows(args.data_dir, "train"))
    heldout = read_rows(args.data_dir, args.split)
    train_wells = {row["metadata"]["well_id"] for row in train}
    heldout_wells = {row["metadata"]["well_id"] for row in heldout}
    assert not train_wells & heldout_wells
    assert all(row["metadata"]["label_provenance"] == "source_direct_folder" for row in train + heldout)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    base.eval()
    results = []
    if args.reuse_base_report:
        previous = json.loads(args.reuse_base_report.read_text(encoding="utf-8"))
        if previous["split"] != args.split or len(previous["records"]) != len(heldout):
            raise ValueError("Base report is from a different split or sample count")
        for row, prior in zip(heldout, previous["records"]):
            meta = row["metadata"]
            if (meta["record_id"], meta["source_archive_member"], meta["well_id"], row["conversations"][1]["value"]) != (
                prior["record_id"], prior["source_archive_member"], prior["well_id"], prior["expected"]
            ):
                raise ValueError("Base report row order or source target differs")
            if normalize(prior["base_raw"]) != prior["base_label"]:
                raise ValueError("Base report label does not match raw answer")
            results.append({key: prior[key] for key in (
                "record_id", "image", "source_archive_member", "well_id", "expected", "base_raw", "base_label"
            )})
    elif not args.skip_base:
        for index, row in enumerate(heldout, 1):
            answer = predict(base, processor, row)
            metadata = row["metadata"]
            results.append({
                "record_id": metadata["record_id"],
                "image": row["image"],
                "source_archive_member": metadata["source_archive_member"],
                "well_id": metadata["well_id"],
                "expected": row["conversations"][1]["value"],
                "base_raw": answer,
                "base_label": normalize(answer),
            })
            if index % 20 == 0:
                print(f"base {index}/{len(heldout)}", flush=True)
    else:
        for row in heldout:
            metadata = row["metadata"]
            results.append({
                "record_id": metadata["record_id"],
                "image": row["image"],
                "source_archive_member": metadata["source_archive_member"],
                "well_id": metadata["well_id"],
                "expected": row["conversations"][1]["value"],
            })
    adapter = PeftModel.from_pretrained(base, args.adapter)
    adapter.eval()
    for index, (row, result) in enumerate(zip(heldout, results), 1):
        answer = predict(adapter, processor, row)
        result["adapter_raw"] = answer
        result["adapter_label"] = normalize(answer)
        if index % 20 == 0:
            print(f"adapter {index}/{len(heldout)}", flush=True)
    by_well = defaultdict(list)
    for result in results:
        by_well[result["well_id"]].append(result)
    report = {
        "scope": "Source-labeled 2x2 cm Mendeley crop lithology and non-core detection, no facies or full descriptions",
        "dataset_url": "https://data.mendeley.com/datasets/b6rj29mrss/1",
        "license": "CC BY 4.0",
        "split": args.split,
        "data_dir": str(args.data_dir.resolve()),
        "adapter": str(args.adapter.resolve()),
        "train_manifest": str(args.train_manifest.resolve()) if args.train_manifest else str((args.data_dir / "train.jsonl").resolve()),
        "reused_base_report": str(args.reuse_base_report.resolve()) if args.reuse_base_report else None,
        "base_predictions_run": not args.skip_base,
        "train_wells": sorted(train_wells),
        "heldout_wells": sorted(heldout_wells),
        "train_crops": len(train),
        "heldout_crops": len(heldout),
        "score": score(results),
        "per_well": {well: score(rows) for well, rows in sorted(by_well.items())},
        "limitations": [
            "Labels come directly from the source folders but are not individually reviewed by a geologist.",
            "Source train/val/test shared all 28 wells; this experiment uses newly separated whole wells.",
            "Test images are neighboring small crops; the number of independent geological units is smaller than the crop count.",
            "No inference about facies, depositional environment, interval boundaries, or full Russian descriptions is justified.",
        ],
        "records": results,
    }
    output = args.output or ROOT / "outputs" / f"mendeley_lithology_pilot_{args.split}_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "score": report["score"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
