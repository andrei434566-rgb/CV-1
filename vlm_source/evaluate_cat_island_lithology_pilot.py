"""Compare base Qwen3-VL and the Cat Island lithology pilot on held-out cores."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import re

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cat_island_lithology_pilot"
MODEL = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTER = ROOT / "outputs" / "core_vlm_cat_island_lithology_pilot_lora"
OUTPUT = ROOT / "outputs" / "cat_island_lithology_pilot_evaluation.json"


def predict(model, processor, row: dict) -> str:
    with Image.open(row["image"]) as opened:
        image = opened.convert("RGB")
        image.thumbnail((448, 448))
    prompt = row["conversations"][0]["value"].replace("<image>", "", 1).strip()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    answer = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return answer.strip()


def lithology(answer: str) -> str | None:
    matched = re.search(r"\{.*\}", answer, re.S)
    if not matched:
        return None
    try:
        result = json.loads(matched.group(0))
        value = result.get("lithology")
        return value.strip().lower() if isinstance(value, str) else None
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("abstain", "four"), default="abstain")
    parser.add_argument("--sample-train", type=int, default=0,
                        help="Diagnostic: run on this many training images instead of held-out images.")
    args = parser.parse_args()
    data = DATA if args.variant == "abstain" else ROOT / "data" / "cat_island_four_class_pilot"
    adapter = ADAPTER if args.variant == "abstain" else ROOT / "outputs" / "core_vlm_cat_island_four_class_pilot_lora"
    manifest = data / ("train.jsonl" if args.sample_train else "heldout.jsonl")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if args.sample_train:
        rows = rows[:args.sample_train]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
    )
    base.eval()
    results = []
    for index, row in enumerate(rows, 1):
        expected = json.loads(row["conversations"][1]["value"])["lithology"].lower()
        answer = predict(base, processor, row)
        results.append({"record_id": row["metadata"]["record_id"],
                        "source_well": row["metadata"]["source_well"],
                        "setting_context": row["metadata"]["setting_context"],
                        "expected": expected, "base_answer": answer, "base_label": lithology(answer)})
        print(f"base {index}/{len(rows)}", flush=True)
    tuned = PeftModel.from_pretrained(base, adapter)
    tuned.eval()
    for index, (row, result) in enumerate(zip(rows, results), 1):
        answer = predict(tuned, processor, row)
        result.update({"adapter_answer": answer, "adapter_label": lithology(answer)})
        print(f"adapter {index}/{len(rows)}", flush=True)
    train_rows = [json.loads(line) for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    train_distribution = Counter(json.loads(row["conversations"][1]["value"])["lithology"].lower() for row in train_rows)
    seen = set(train_distribution)
    majority = train_distribution.most_common(1)[0][0]
    for result in results:
        result["label_seen_in_train"] = result["expected"] in seen
    summary = {"heldout_images": len(results),
               "heldout_cores": sorted({row["source_well"] for row in results}),
               "seen_label_images": sum(row["label_seen_in_train"] for row in results),
               "base_correct": sum(row["base_label"] == row["expected"] for row in results),
               "adapter_correct": sum(row["adapter_label"] == row["expected"] for row in results),
               "train_majority_label": majority,
               "majority_baseline_correct": sum(row["expected"] == majority for row in results),
               "base_correct_seen_labels": sum(row["label_seen_in_train"] and row["base_label"] == row["expected"] for row in results),
               "adapter_correct_seen_labels": sum(row["label_seen_in_train"] and row["adapter_label"] == row["expected"] for row in results),
               "expected_distribution": dict(Counter(row["expected"] for row in results)),
               "base_distribution": dict(Counter(row["base_label"] or "invalid" for row in results)),
               "adapter_distribution": dict(Counter(row["adapter_label"] or "invalid" for row in results)),
               "limitations": ["Source lithology labels and automatically cropped images were not approved by a geologist.",
                               "Accuracy is exact Russian class match on automatically cropped core intervals, not full-description or facies accuracy.",
                               "Forced-choice results are diagnostic only and do not measure safe abstention." if args.variant == "four" else "The prompt allows an uncertain answer."]}
    if args.variant == "four":
        output = ROOT / "outputs" / ("cat_island_four_class_train_diagnostic.json" if args.sample_train else "cat_island_four_class_evaluation.json")
    else:
        output = OUTPUT if not args.sample_train else ROOT / "outputs" / "cat_island_lithology_pilot_train_diagnostic.json"
    output.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
