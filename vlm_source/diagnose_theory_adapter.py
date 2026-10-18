"""Compare teacher-forced loss and generated text on an exact training row."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "kern_analyzer_app" / "app" / "infrastructure" / "ml"))
from qwen_train import CorePhotoDataset  # noqa: E402

BASE = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTER = ROOT / "outputs" / "core_vlm_theory_v4b_focused_lora"
MANIFEST = ROOT / "data" / "knowledge" / "theory_training_v4_focused.jsonl"
SYSTEM = (ROOT / "data" / "knowledge" / "theory_system_v4.txt").read_text(encoding="utf-8").strip()


def main() -> None:
    row = json.loads(MANIFEST.read_text(encoding="utf-8").splitlines()[1])
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    sample = CorePhotoDataset([row], processor, 448, SYSTEM)[0]
    prompt_length = int((sample["labels"] == -100).sum())
    prompt_ids = sample["input_ids"][:prompt_length].unsqueeze(0).to("cuda")
    full = {key: value.unsqueeze(0).to("cuda") for key, value in sample.items()}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float16, attn_implementation="sdpa",
        device_map="cuda", local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    with torch.inference_mode():
        forward = model(**full)
        loss = float(forward.loss)
        first_logits = forward.logits[0, prompt_length - 1]
        first_target_id = int(sample["input_ids"][prompt_length])
        first_target_probability = float(first_logits.softmax(dim=-1)[first_target_id])
        top_ids = first_logits.topk(5).indices.tolist()
        result = model.generate(input_ids=prompt_ids, max_new_tokens=120, do_sample=False)
    report = {
        "row": row["metadata"]["record_id"],
        "teacher_forced_loss": loss,
        "first_target_token": processor.tokenizer.decode([first_target_id]),
        "first_target_probability": first_target_probability,
        "first_token_top5": [processor.tokenizer.decode([token_id]) for token_id in top_ids],
        "reference": row["conversations"][1]["value"],
        "generated": processor.batch_decode(result[:, prompt_length:], skip_special_tokens=True)[0],
        "active_adapter": str(model.active_adapter),
        "prompt_suffix": processor.tokenizer.decode(prompt_ids[0, -35:]),
    }
    output = ROOT / "outputs" / "theory_v4_evaluation" / "teacher_forced_diagnostic.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"teacher_forced_loss={loss:.4f}; report={output}")


if __name__ == "__main__":
    main()
