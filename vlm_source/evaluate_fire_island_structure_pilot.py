"""Run a Qwen adapter on fully unseen Fire Island cores (diagnostic only).

Reports JSON/depth formatting and raw descriptions. Geological agreement must
be assessed against images and official logs; keywords are not accuracy.
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
MANIFEST = ROOT / "data/usgs_fire_island_structure_pilot/heldout.jsonl"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 8 or {r["metadata"]["core_id"] for r in records} != {"P25-C05", "P29-C11"}:
        raise ValueError("Expected eight windows from the two reserved cores")
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
            generated = model.generate(**inputs, max_new_tokens=220, do_sample=False)
        completion = generated[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(completion, skip_special_tokens=True)[0].strip()
        if answer.startswith("```"):
            answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        target = json.loads(record["conversations"][1]["value"])["description_segments"][0]
        try:
            parsed = json.loads(answer)
            segments = parsed.get("description_segments", [])
            valid_schema = isinstance(segments, list) and len(segments) == 1
            segment = segments[0] if valid_schema else {}
            valid_schema = (valid_schema and isinstance(segment.get("description"), str)
                            and bool(segment["description"].strip())
                            and isinstance(segment.get("top_m"), (float, int))
                            and isinstance(segment.get("base_m"), (float, int)))
            description = segment["description"] if valid_schema else answer
            correct_depth = (valid_schema and abs(segment["top_m"] - target["top_m"]) < .001
                             and abs(segment["base_m"] - target["base_m"]) < .001)
        except (ValueError, TypeError, KeyError, AttributeError):
            valid_schema = correct_depth = False
            description = answer
        row = {
            "id": record["metadata"]["record_id"], "core_id": record["metadata"]["core_id"],
            "image": record["image"], "source_window_cm": record["metadata"]["window_cm"],
            "source_description_en": record["metadata"]["source_description_en"],
            "expected_visual_draft": target["description"],
            "valid_one_segment_json": bool(valid_schema), "exact_input_depths": bool(correct_depth),
            "sand_word_in_raw_output": bool(re.search(r"пес[оакчн]", answer.lower())),
            "output": answer,
        }
        results.append(row)
        print(f"{i}/{len(records)} {row['id']}: schema={valid_schema} depth={correct_depth} "
              f"sand={row['sand_word_in_raw_output']} {description[:110]}", flush=True)
    result = {
        "adapter": str(args.adapter.resolve()),
        "heldout_cores": sorted({row["core_id"] for row in results}),
        "n": len(results),
        "valid_schema_count": sum(row["valid_one_segment_json"] for row in results),
        "exact_depth_count": sum(row["exact_input_depths"] for row in results),
        "sand_word_count": sum(row["sand_word_in_raw_output"] for row in results),
        "metric_warning": "Formatting and word presence are not geological accuracy; all Russian targets are experimental drafts.",
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
