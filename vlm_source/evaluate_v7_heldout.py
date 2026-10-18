"""Evaluate R-41 held-out photo blocks with identical decoding settings.

Scores are lexical proxies, not expert geological validation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from core_sedimentologist_app import extract_description


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
MANIFEST = ROOT / "data" / "v7_heldout" / "test_visual.jsonl"
ROCKS = {
    "песчаник": r"песчаник\w*",
    "алевролит": r"алевролит\w*",
    "аргиллит": r"аргиллит\w*",
    "глина": r"\bглин[аыиеою]\w*|глинист\w*",
    "уголь": r"\bугол[ья]\w*|углист\w*",
    "известняк": r"известняк\w*",
    "доломит": r"доломит\w*",
    "мергель": r"мергел\w*",
}
COLORS = {
    "серый": r"сер\w*",
    "тёмный": r"т[её]мн\w*",
    "светлый": r"светл\w*",
    "чёрный": r"ч[её]рн\w*",
    "бурый": r"бур\w*|коричнев\w*",
    "зелёный": r"зел[её]н\w*",
    "жёлтый": r"ж[её]лт\w*",
    "красный": r"красн\w*|рыж\w*",
    "белый": r"бел\w*",
}


def tags(text: str, patterns: dict[str, str]) -> set[str]:
    return {tag for tag, pattern in patterns.items() if re.search(pattern, text.lower())}


def f1(counts: Counter) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    value = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": value}


def evaluate(rows: list[dict]) -> dict:
    rock, color = Counter(), Counter()
    for row in rows:
        for patterns, accumulator in [(ROCKS, rock), (COLORS, color)]:
            expected = tags(row["reference_description"], patterns)
            predicted = tags(row["predicted_description"], patterns)
            accumulator["tp"] += len(expected & predicted)
            accumulator["fp"] += len(predicted - expected)
            accumulator["fn"] += len(expected - predicted)
    return {
        "representative_rows": len(rows),
        "nonempty_description": sum(bool(row["predicted_description"]) for row in rows),
        "valid_json": sum(row["valid_json"] for row in rows),
        "rock_token_micro": f1(rock),
        "color_token_micro": f1(color),
        "caution": "Russian keyword overlap with the first source sentence only. Synonyms, image visibility, and false visual claims require a geologist; these numbers are not facies accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_row = defaultdict(list)
    for row in records:
        by_row[row["metadata"]["source_row"]].append(row)
    chosen = [sorted(group, key=lambda row: row["metadata"]["photo_tile_depth_interval_m"][0])[len(group) // 2] for _, group in sorted(by_row.items())]
    processor = AutoProcessor.from_pretrained(BASE)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for start in range(0, len(chosen), args.batch_size):
            batch = chosen[start:start + args.batch_size]
            images, prompts = [], []
            for row in batch:
                with Image.open(row["image"]) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((448, 448))
                images.append(image)
                prompt = row["conversations"][0]["value"].replace("<image>\n", "", 1)
                message = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                prompts.append(processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, repetition_penalty=1.12, no_repeat_ngram_size=4,
                )
            raw_outputs = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch, raw_outputs, strict=True):
                description, valid_json = extract_description(raw)
                reference = json.loads(row["conversations"][1]["value"])["description_segments"][0]["description"]
                result = {
                    "record_id": row["metadata"]["record_id"],
                    "source_row": row["metadata"]["source_row"],
                    "image": row["image"],
                    "interval_m": row["metadata"]["photo_tile_depth_interval_m"],
                    "reference_description": reference,
                    "source_description_full": row["metadata"]["source_description_full"],
                    "predicted_description": description,
                    "valid_json": valid_json,
                    "raw": raw,
                    "model": args.label,
                }
                results.append(result)
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(f"{args.label}: {len(results)}/{len(chosen)}", flush=True)
    summary = evaluate(results)
    summary.update({"model": args.label, "adapter": str(args.adapter) if args.adapter else None,
                    "holdout": str(args.manifest), "prediction_file": str(args.output)})
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
