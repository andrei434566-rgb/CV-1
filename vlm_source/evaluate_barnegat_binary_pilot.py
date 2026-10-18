"""Compare base Qwen and the Barnegat binary pilot on whole held-out cores."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "barnegat_binary_pilot"
MODEL = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTER = ROOT / "outputs" / "core_vlm_barnegat_binary_pilot_lora"
OUTPUT = ROOT / "outputs" / "barnegat_binary_pilot_evaluation.json"
LABELS = ("песчаный", "тонкозернистый")


def read_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def predict(model, processor, row: dict) -> str:
    with Image.open(row["image"]) as opened:
        image = opened.convert("RGB")
        image.thumbnail((448, 448))
    prompt = row["conversations"][0]["value"].replace("<image>", "", 1).strip()
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def normalize(answer: str) -> str | None:
    found = [label for label in LABELS if re.search(rf"(?<!\w){label}(?!\w)", answer.lower())]
    return found[0] if len(found) == 1 else None


def macro_f1(rows: list[dict], key: str) -> float:
    scores = []
    for label in LABELS:
        tp = sum(r["expected"] == label and r[key] == label for r in rows)
        fp = sum(r["expected"] != label and r[key] == label for r in rows)
        fn = sum(r["expected"] == label and r[key] != label for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def interval_summary(results: list[dict]) -> dict:
    """Aggregate repeated tiles so a long logged interval counts once."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        groups[result["record_id"].split("_tile", 1)[0]].append(result)
    collapsed = []
    for interval_id, tiles in groups.items():
        row = {"interval_id": interval_id, "well_id": tiles[0]["well_id"], "expected": tiles[0]["expected"], "tiles": len(tiles)}
        for mode in ("base", "adapter"):
            counts = Counter(tile[f"{mode}_label"] for tile in tiles)
            best = counts.most_common()
            row[f"{mode}_label"] = best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None
        collapsed.append(row)
    return {
        "count": len(collapsed),
        "base_correct": sum(r["base_label"] == r["expected"] for r in collapsed),
        "adapter_correct": sum(r["adapter_label"] == r["expected"] for r in collapsed),
        "base_macro_f1": macro_f1(collapsed, "base_label"),
        "adapter_macro_f1": macro_f1(collapsed, "adapter_label"),
        "expected_distribution": dict(Counter(r["expected"] for r in collapsed)),
        "per_well": {
            well: {
                "intervals": sum(r["well_id"] == well for r in collapsed),
                "base_correct": sum(r["well_id"] == well and r["base_label"] == r["expected"] for r in collapsed),
                "adapter_correct": sum(r["well_id"] == well and r["adapter_label"] == r["expected"] for r in collapsed),
            }
            for well in sorted({r["well_id"] for r in collapsed})
        },
        "records": collapsed,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    train = read_rows("train_candidates.jsonl")
    holdout = read_rows("holdout_candidates.jsonl")
    train_wells = {r["metadata"]["well_id"] for r in train}
    holdout_wells = {r["metadata"]["well_id"] for r in holdout}
    if train_wells & holdout_wells:
        raise ValueError("Train/holdout well overlap")
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
    )
    model.eval()
    results = []
    for index, row in enumerate(holdout, 1):
        answer = predict(model, processor, row)
        results.append({
            "record_id": row["metadata"]["record_id"],
            "well_id": row["metadata"]["well_id"],
            "image": row["image"],
            "source_lithology": row["metadata"]["source_lithology"],
            "expected": row["conversations"][1]["value"],
            "base_raw": answer,
            "base_label": normalize(answer),
        })
        print(f"base {index}/{len(holdout)}", flush=True)
    tuned = PeftModel.from_pretrained(model, ADAPTER)
    tuned.eval()
    for index, (row, result) in enumerate(zip(holdout, results), 1):
        answer = predict(tuned, processor, row)
        result["adapter_raw"] = answer
        result["adapter_label"] = normalize(answer)
        print(f"adapter {index}/{len(holdout)}", flush=True)
    majority = Counter(r["conversations"][1]["value"] for r in train).most_common(1)[0][0]
    summary = {
        "heldout_images": len(holdout),
        "train_wells": sorted(train_wells),
        "heldout_wells": sorted(holdout_wells),
        "base_correct": sum(r["base_label"] == r["expected"] for r in results),
        "adapter_correct": sum(r["adapter_label"] == r["expected"] for r in results),
        "majority_label": majority,
        "majority_correct": sum(r["expected"] == majority for r in results),
        "base_macro_f1": macro_f1(results, "base_label"),
        "adapter_macro_f1": macro_f1(results, "adapter_label"),
        "expected_distribution": dict(Counter(r["expected"] for r in results)),
        "base_distribution": dict(Counter(r["base_label"] or "неопределено" for r in results)),
        "adapter_distribution": dict(Counter(r["adapter_label"] or "неопределено" for r in results)),
        "source_interval_evaluation": interval_summary(results),
        "scope": "Two-class sand-versus-fine pilot; not full core description or depositional environment accuracy.",
        "limitations": [
            "Pixel/depth alignment and binary class mapping are automatic and have not been approved by a geologist.",
            "Multiple tiles from the same logged interval are correlated; the split is by whole core, not by tile.",
            "All cores come from Barnegat Bay; external geographic and ancient-core validation is still absent.",
        ],
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
