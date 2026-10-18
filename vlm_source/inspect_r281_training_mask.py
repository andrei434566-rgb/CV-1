"""Inspect which tokens the current VLM trainer actually supervises."""

import json
from pathlib import Path

from transformers import AutoProcessor

from train_qwen3vl import CorePhotoDataset


root = Path(__file__).resolve().parent
model = root / "model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
rows = [json.loads(line) for line in (root / "data/r281_lithology_experiment/train.jsonl").read_text(encoding="utf-8").splitlines() if line]
processor = AutoProcessor.from_pretrained(model)
dataset = CorePhotoDataset(rows, processor, 448)
for index in [0, 15, 75]:
    item = dataset[index]
    ids = item["input_ids"]
    labels = item["labels"]
    supervised = ids[labels != -100]
    print(json.dumps({
        "index": index,
        "expected": rows[index]["conversations"][1]["value"],
        "total_tokens": len(ids),
        "supervised_tokens": len(supervised),
        "supervised_text": processor.tokenizer.decode(supervised),
    }, ensure_ascii=False))
