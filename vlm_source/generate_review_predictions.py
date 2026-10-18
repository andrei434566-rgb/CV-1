"""Generate v5 draft descriptions for the confirmed R-41 interval crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = (
    PROJECT_ROOT
    / "model_cache"
    / "hub"
    / "models--Qwen--Qwen3-VL-2B-Instruct"
    / "snapshots"
    / "89644892e4d85e24eaac8bacfd4f463576704203"
)
DEFAULT_ADAPTER = PROJECT_ROOT / "outputs" / "core_vlm_v5_interval_crops_lora"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "e_drive_interval_crops_v2" / "qwen3_vl_interval_crops.jsonl"
DEFAULT_OUTPUT = DEFAULT_ADAPTER / "geologist_review_predictions.jsonl"


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--one-per-source-interval",
        action="store_true",
        help="Keep only the first image crop for each source photo-description row.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for this model.")

    rows = [
        row for row in load_rows(args.manifest)
        if row.get("metadata", {}).get("record_type") == "user_confirmed_interval_crop"
    ]
    if args.one_per_source_interval:
        representatives = []
        seen_intervals = set()
        for row in rows:
            metadata = row.get("metadata", {})
            key = (metadata.get("source_photo_record_id"), metadata.get("source_row"))
            if key not in seen_intervals:
                representatives.append(row)
                seen_intervals.add(key)
        rows = representatives
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if args.output.exists():
        for record in load_rows(args.output):
            existing[record["record_id"]] = record

    processor = AutoProcessor.from_pretrained(args.base_model)
    # Decoder-only generation uses left padding for correctly aligned batched prompts.
    processor.tokenizer.padding_side = "left"
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    pending = [row for row in rows if row.get("metadata", {}).get("record_id") not in existing]
    print(f"Visual records: {len(rows)}; already generated: {len(rows) - len(pending)}; pending: {len(pending)}")
    with args.output.open("a", encoding="utf-8", newline="\n") as output_handle:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            images = []
            texts = []
            for row in batch:
                image_path = Path(row["image"])
                if not image_path.is_absolute():
                    image_path = PROJECT_ROOT / image_path
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((args.max_image_side, args.max_image_side))
                images.append(image)
                user_prompt = row["conversations"][0]["value"]
                if user_prompt.startswith("<image>\n"):
                    user_prompt = user_prompt[len("<image>\n") :]
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt},
                    ],
                }]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            completion = generated[:, inputs["input_ids"].shape[1] :]
            decoded = processor.batch_decode(completion, skip_special_tokens=True)
            for row, raw in zip(batch, decoded, strict=True):
                metadata = row.get("metadata", {})
                record_id = metadata["record_id"]
                raw = raw.strip()
                parsed = None
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    if raw.startswith("```"):
                        candidate = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                        try:
                            parsed = json.loads(candidate)
                            raw = candidate
                        except json.JSONDecodeError:
                            pass

                segments = parsed.get("description_segments", []) if isinstance(parsed, dict) else []
                draft = ""
                if segments and isinstance(segments[0], dict):
                    draft = str(segments[0].get("description", ""))
                result = {
                    "record_id": record_id,
                    "image": row["image"],
                    "crop_interval_m": metadata.get("photo_tile_depth_interval_m"),
                    "source_description_interval_m": metadata.get("source_description_interval_m"),
                    "source_photo_record_id": metadata.get("source_photo_record_id"),
                    "source_row": metadata.get("source_row"),
                    "source_workbook": metadata.get("source_workbook"),
                    "source_description_full": metadata.get("source_description_full", ""),
                    "reference_target_json": row["conversations"][1]["value"],
                    "v5_draft_description": draft,
                    "v5_output_raw": raw,
                    "valid_json": isinstance(parsed, dict),
                    "adapter": str(args.adapter),
                    "inference_prompt_matches_training": True,
                }
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                existing[record_id] = result
                print(f"[{len(existing)}/{len(rows)}] {record_id}: {draft or '[JSON parse failed]'}")
            output_handle.flush()


if __name__ == "__main__":
    main()
