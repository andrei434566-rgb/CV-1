"""Compare base Qwen3-VL with DS1061 research LoRA on held-out sites."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "usgs_cache_creek_ds1061_binary_experiment"
MODEL = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTER = ROOT / "outputs" / "core_vlm_ds1061_binary_research_lora"
OUTPUT = ROOT / "outputs" / "usgs_ds1061_binary_research_evaluation.json"
LABELS = ("песчаный", "глинистый")


def read_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def predict(model, processor, row: dict) -> str:
    with Image.open(row["image"]) as source:
        image = source.convert("RGB")
        image.thumbnail((448, 448))
    prompt = row["conversations"][0]["value"].replace("<image>", "", 1).strip()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def normalize(raw: str) -> str | None:
    found = [label for label in LABELS if re.search(rf"(?<!\w){label}(?!\w)", raw.casefold())]
    return found[0] if len(found) == 1 else None


def score(rows: list[dict], key: str) -> dict:
    result = {
        "count": len(rows),
        "correct": sum(r[key] == r["expected"] for r in rows),
        "expected": dict(Counter(r["expected"] for r in rows)),
        "predicted": dict(Counter(r[key] or "неопределено" for r in rows)),
    }
    f1 = []
    for label in LABELS:
        tp = sum(r["expected"] == label and r[key] == label for r in rows)
        fp = sum(r["expected"] != label and r[key] == label for r in rows)
        fn = sum(r["expected"] == label and r[key] != label for r in rows)
        result[label] = {"tp": tp, "fp": fp, "fn": fn, "recall": tp / (tp + fn) if tp + fn else None}
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    result["macro_f1"] = sum(f1) / len(f1)
    return result


def main() -> None:
    train = read_rows("train_experimental.jsonl")
    heldout = read_rows("heldout.jsonl")
    train_sites = {r["metadata"]["site_group"] for r in train}
    heldout_sites = {r["metadata"]["site_group"] for r in heldout}
    if train_sites & heldout_sites or heldout_sites != {"2", "5", "12"}:
        raise ValueError("Site-level split is invalid")
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
    )
    base.eval()
    results = []
    for i, row in enumerate(heldout, 1):
        raw = predict(base, processor, row)
        results.append({
            "record_id": row["metadata"]["record_id"],
            "site_group": row["metadata"]["site_group"],
            "well_id": row["metadata"]["well_id"],
            "image": row["image"],
            "source_label_en": row["metadata"]["source_label_en"],
            "expected": row["conversations"][1]["value"],
            "base_raw": raw,
            "base_label": normalize(raw),
        })
        if i % 5 == 0 or i == len(heldout):
            print(f"base {i}/{len(heldout)}", flush=True)
    tuned = PeftModel.from_pretrained(base, ADAPTER)
    tuned.eval()
    for i, (row, result) in enumerate(zip(heldout, results), 1):
        raw = predict(tuned, processor, row)
        result["adapter_raw"] = raw
        result["adapter_label"] = normalize(raw)
        if i % 5 == 0 or i == len(heldout):
            print(f"adapter {i}/{len(heldout)}", flush=True)
    majority = Counter(r["conversations"][1]["value"] for r in train).most_common(1)[0][0]
    summary = {
        "scope": "Research-only binary source lithology classification, NOT full description or facies accuracy.",
        "train_sites": sorted(train_sites, key=int),
        "heldout_sites": sorted(heldout_sites, key=int),
        "heldout_unique_images": len(heldout),
        "base": score(results, "base_label"),
        "adapter": score(results, "adapter_label"),
        "majority_label_in_oversampled_train": majority,
        "majority_correct_on_holdout": sum(r["expected"] == majority for r in results),
        "per_site": {
            site: {
                "base": score([r for r in results if r["site_group"] == site], "base_label"),
                "adapter": score([r for r in results if r["site_group"] == site], "adapter_label"),
            }
            for site in sorted(heldout_sites, key=int)
        },
        "limitations": [
            "Source PDF remarks were only approximately mapped to image centers and have no geologist visual approval.",
            "Some macroscopic sand/clay distinctions may be impossible from the narrow image alone.",
            "All sites are from one modern settling basin; no ancient-core or depositional-setting accuracy follows.",
        ],
    }
    OUTPUT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"base": summary["base"], "adapter": summary["adapter"], "majority_correct": summary["majority_correct_on_holdout"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
