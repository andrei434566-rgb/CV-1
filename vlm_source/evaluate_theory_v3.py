"""Generate blinded, text-only answers to the held-out v3 theory questions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
HOLDOUT = ROOT / "data" / "knowledge" / "theory_holdout_v3.jsonl"
SYSTEM = (
    "Ты локальный ИИ-помощник Kern Analyzer. Отвечай по-русски как помощник геолога-седиментолога. "
    "Литотип означает разновидность горной породы, фация — геологическую осадочную фацию, "
    "биотурбация — переработку осадка организмами. Сначала называй диагностические признаки, "
    "затем пределы уверенности. Не выдумывай свойства, не видимые на фотографии."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=170)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    system = (args.system_prompt_file.read_text(encoding="utf-8").strip()
              if args.system_prompt_file else SYSTEM)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the local Qwen evaluation")
    if not (BASE / "config.json").is_file():
        raise FileNotFoundError(BASE)
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float16, attn_implementation="sdpa",
        device_map="cuda", local_files_only=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows = [json.loads(line) for line in args.holdout.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]
    output = []
    for row in rows:
        question = row["conversations"][0]["value"]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0].strip()
        output.append({
            "record_id": row["metadata"]["record_id"],
            "question": question,
            "reference_answer": row["conversations"][1]["value"],
            "generated_answer": answer,
            "sources": row["metadata"]["sources"],
        })
        print(row["metadata"]["record_id"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": str(args.adapter) if args.adapter else str(BASE),
        "prompt": system,
        "holdout": str(args.holdout),
        "assessment": "Qualitative held-out theory questions; no automatic accuracy or image score.",
        "rows": output,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
