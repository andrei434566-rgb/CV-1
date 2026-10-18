"""Generate blinded text-only comparison answers on four held-out theory questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
HOLDOUT = ROOT / "data" / "knowledge" / "theory_holdout_v2.jsonl"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    processor = AutoProcessor.from_pretrained(BASE)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows = [json.loads(line) for line in HOLDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in rows:
        question = row["conversations"][0]["value"]
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": question}]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = processor.batch_decode(
            tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        output.append({
            "record_id": row["metadata"]["record_id"],
            "question": question,
            "reference_answer": row["conversations"][1]["value"],
            "generated_answer": answer,
            "source": row["metadata"]["sources"][0],
        })
        print(row["metadata"]["record_id"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": str(args.adapter) if args.adapter else str(BASE),
        "assessment": "Qualitative held-out text-only comparison; no automatic accuracy score or photo test.",
        "rows": output,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
