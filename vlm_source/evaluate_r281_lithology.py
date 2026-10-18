"""Evaluate cut-face lithology on entire held-out R-281 cores."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
TEST = ROOT / "data" / "r281_lithology_experiment" / "test.jsonl"
CLASSES = ("песчаник", "алевролит", "аргиллит", "уголь", "переслаивание")


def parse_class(raw: str) -> tuple[str, bool]:
    cleaned = raw.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[len("```json"): -3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    try:
        value = json.loads(cleaned)
        candidate = str(value.get("rock_type", "")).lower().strip()
        valid_json = True
    except (ValueError, AttributeError, TypeError):
        candidate = raw.lower()
        valid_json = False
    if candidate in CLASSES:
        return candidate, valid_json
    stems = {
        "песчаник": r"песчаник\w*",
        "алевролит": r"алевролит\w*",
        "аргиллит": r"аргиллит\w*",
        "уголь": r"угол[ья]\w*|угл[ьея]\w*",
        "переслаивание": r"переслаиван\w*",
    }
    matches = [name for name, pattern in stems.items() if re.search(r"\b(?:" + pattern + r")\b", candidate)]
    if len(matches) == 1:
        return matches[0], valid_json
    return "неопределено", valid_json


def score(rows: list[dict]) -> dict:
    labels = sorted({row["expected"] for row in rows})
    per_class = {}
    for label in labels:
        tp = sum(row["expected"] == label and row["predicted"] == label for row in rows)
        fp = sum(row["expected"] != label and row["predicted"] == label for row in rows)
        fn = sum(row["expected"] == label and row["predicted"] != label for row in rows)
        per_class[label] = {"support": tp + fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "recall": tp / (tp + fn) if tp + fn else 0.0,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    confusion = {label: dict(Counter(row["predicted"] for row in rows if row["expected"] == label)) for label in labels}
    return {"n": len(rows), "accuracy": sum(row["expected"] == row["predicted"] for row in rows) / len(rows),
            "correct": sum(row["expected"] == row["predicted"] for row in rows),
            "macro_f1": sum(value["f1"] for value in per_class.values()) / len(per_class),
            "valid_json": sum(row["valid_json"] for row in rows),
            "abstentions": sum(row["predicted"] == "неопределено" for row in rows),
            "per_class": per_class, "confusion": confusion}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=TEST)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rescore-existing", action="store_true", help="Re-score saved generations without loading a model")
    args = parser.parse_args()
    if args.rescore_existing:
        results = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line]
        for result in results:
            result["predicted"], result["valid_json"] = parse_class(result["raw"])
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
        summary = score(results)
        summary.update({"label": args.label, "adapter": str(args.adapter) if args.adapter else None,
                        "test_cores": [2, 9], "one_well": "281Р",
                        "caution": "Class labels are derived from adjacent report text; images are low-resolution and blue-tinted. This is not facies or full-description accuracy."})
        args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        records = records[:args.limit]
    processor = AutoProcessor.from_pretrained(BASE)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16,
                                                            attn_implementation="sdpa", device_map="cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w", encoding="utf-8") as target:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            images, prompts = [], []
            for row in batch:
                with Image.open(row["image"]) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((448, 448))
                images.append(image)
                prompt = row["conversations"][0]["value"].replace("<image>\n", "", 1)
                message = [{"role": "user", "content": [{"type": "image", "image": image},
                                                       {"type": "text", "text": prompt}]}]
                prompts.append(processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                           repetition_penalty=1.1, no_repeat_ngram_size=4)
            raw_outputs = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch, raw_outputs, strict=True):
                predicted, valid_json = parse_class(raw)
                result = {"id": row["metadata"]["record_id"], "core": row["metadata"]["core_number"],
                          "image": row["image"], "expected": row["metadata"]["rock_type"],
                          "predicted": predicted, "valid_json": valid_json, "raw": raw,
                          "source_description": row["metadata"]["source_description"], "model": args.label}
                results.append(result)
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(f"{args.label}: {len(results)}/{len(records)}", flush=True)
    summary = score(results)
    summary.update({"label": args.label, "adapter": str(args.adapter) if args.adapter else None,
                    "manifest": str(args.manifest),
                    "core_numbers": sorted({row["core"] for row in results}), "one_well": "281Р",
                    "caution": "Class labels are derived from adjacent report text; images are low-resolution and blue-tinted. This is not facies or full-description accuracy."})
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
