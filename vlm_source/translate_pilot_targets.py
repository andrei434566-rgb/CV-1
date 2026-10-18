"""Translate geology-approved English pilot targets into Russian with cached local Qwen."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
MANIFEST = ROOT / "data" / "qwen3_vl_v3_reviewed_pilot.jsonl"
OUTPUT = ROOT / "data" / "translation_candidates_pilot_01.jsonl"


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def translate_description(processor, model, text: str) -> str:
    prompt = (
        "You are a professional sedimentologist translating a field log. Translate the complete text "
        "into accurate, natural Russian. Do not summarize, omit, or add observations or interpretation. "
        "Preserve every number, Munsell code, range, percentage, and distinction in the source. "
        "Translate lithology terms: sand=песок; mud=ил; sandy mud=песчанистый ил; muddy sand=илистый песок; "
        "silt=алеврит; clay=глина; peat=торф. Translate color names into Russian and keep their original codes. "
        "Translate shell lag as раковинный лаг (скопление раковинного материала). Output only the Russian text.\n\n"
        + text
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=min(768, max(180, len(text) // 2 + 120)),
            do_sample=False,
        )
    completion = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(completion, skip_special_tokens=True)[0].strip().strip('"“”')


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for local translation.")

    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("metadata", {}).get("dataset")]
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()

    outputs = []
    for index, record in enumerate(rows, start=1):
        original_text = record["conversations"][1]["value"]
        source = json.loads(original_text)
        original_segments = source.get("description_segments", [])
        translated_segments = []
        for segment in original_segments:
            translated_segments.append(
                {
                    "top_m": segment.get("top_m"),
                    "base_m": segment.get("base_m"),
                    "description": translate_description(processor, model, segment["description"]),
                }
            )
        translated = {**source, "description_segments": translated_segments}
        translated_text = json.dumps(translated, ensure_ascii=False)
        source_numbers = [re.sub(r",", ".", value) for value in re.findall(r"\d+(?:[.,]\d+)?", " ".join(s["description"] for s in original_segments))]
        translated_numbers = [re.sub(r",", ".", value) for value in re.findall(r"\d+(?:[.,]\d+)?", " ".join(s["description"] for s in translated_segments))]
        all_has_cyrillic = all(re.search(r"[а-яё]", s["description"], flags=re.IGNORECASE) for s in translated_segments)
        source_descriptions = " ".join(s["description"] for s in original_segments)
        translated_descriptions = " ".join(s["description"] for s in translated_segments)
        outputs.append(
            {
                "record_id": record["metadata"]["record_id"],
                "dataset": record["metadata"].get("dataset"),
                "core_key": record["metadata"].get("core_key"),
                "image": record.get("image"),
                "source_target": original_text,
                "proposed_russian_target": translated_text,
                "source_target_sha256": hashlib.sha256(original_text.encode("utf-8")).hexdigest(),
                "translation_model": "Qwen3-VL-2B-Instruct (local, deterministic)",
                "translation_qa": {
                    "segment_count_preserved": len(original_segments) == len(translated_segments),
                    "depth_boundaries_preserved": all(
                        old.get(key) == new.get(key)
                        for old, new in zip(original_segments, translated_segments, strict=True)
                        for key in ("top_m", "base_m")
                    ),
                    "numeric_tokens_preserved": source_numbers == translated_numbers,
                    "russian_text_in_every_segment": all_has_cyrillic,
                    "source_and_translation_differ": source_descriptions != translated_descriptions,
                },
                "review_status": "translation_pending_user_review",
            }
        )
        OUTPUT.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in outputs) + "\n", encoding="utf-8")
        print(f"translated {index}/{len(rows)}: {record['metadata']['record_id']}; qa={outputs[-1]['translation_qa']}", flush=True)

    print(f"saved={OUTPUT}; rows={len(outputs)}")


if __name__ == "__main__":
    main()
