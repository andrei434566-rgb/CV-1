"""Evaluate one Qwen adapter on core-disjoint Midnight Pass windows.

This is a descriptive diagnostic. Keyword checks are not geological accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
MODEL = (ROOT / "model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
         "89644892e4d85e24eaac8bacfd4f463576704203")
MANIFEST = ROOT / "data/usgs_midnight_pass_shell_pilot/heldout.jsonl"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    results = []
    for i, record in enumerate(records, 1):
        with Image.open(record["image"]) as opened:
            image = opened.convert("RGB")
            image.thumbnail((448, 448))
        prompt = record["conversations"][0]["value"].replace("<image>", "", 1).strip()
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": prompt}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=180, do_sample=False)
        completion = generated[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(completion, skip_special_tokens=True)[0].strip()
        if answer.startswith("```"):
            answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        target = json.loads(record["conversations"][1]["value"])["description_segments"][0]
        try:
            parsed = json.loads(answer)
            segments = parsed.get("description_segments", [])
            valid_json = isinstance(segments, list) and len(segments) == 1
            segment = segments[0] if valid_json else {}
            description = str(segment.get("description", ""))
            correct_depth = (valid_json and abs(float(segment.get("top_m", -1)) - target["top_m"]) < .001
                             and abs(float(segment.get("base_m", -1)) - target["base_m"]) < .001)
        except (ValueError, TypeError, KeyError, AttributeError):
            valid_json = correct_depth = False
            description = answer
        shell_mentioned = bool(re.search(r"раковин|ракуш|створк", answer.lower()))
        row = {"id": record["metadata"]["record_id"],
               "core_id": record["metadata"]["core_id"],
               "source_lithology_en": record["metadata"]["source_lithology_en"],
               "image": record["image"], "source_window_cm": record["metadata"]["window_cm"],
               "valid_one_segment_json": valid_json, "exact_input_depths": correct_depth,
               "shell_word_in_raw_output": shell_mentioned,
               "output": answer, "expected_visual_target": target["description"]}
        results.append(row)
        print(f"{i}/{len(records)} {row['id']}: json={valid_json} depth={correct_depth} "
              f"shell={shell_mentioned} {description[:100]}", flush=True)
    result = {"adapter": str(args.adapter.resolve()),
              "heldout_cores": sorted(set(row["core_id"] for row in results)),
              "n": len(results),
              "valid_json_count": sum(row["valid_one_segment_json"] for row in results),
              "exact_depth_count": sum(row["exact_input_depths"] for row in results),
              "shell_word_in_raw_output_count": sum(row["shell_word_in_raw_output"] for row in results),
              "metric_warning": "Keyword overlap is not evidence of correct geological description.",
              "records": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
