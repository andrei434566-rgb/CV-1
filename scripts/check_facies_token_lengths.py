"""Audit exact Qwen SFT chat-template lengths on CPU before allocating the GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_qwen_source_pilot import _deny_network, read_records, record_id, validate_training_tokens


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "open_datasets/qwen_facies_reasoning_v1_v2")
    parser.add_argument("--model", type=Path, default=ROOT.parent / "vlm_core_v1/model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203")
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--prior-observed-tokens", type=int, default=717)
    parser.add_argument("--output", type=Path, help="Optional report path outside the frozen dataset")
    args = parser.parse_args()
    if args.max_sequence_tokens <= 0 or args.prior_observed_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if args.output and args.output.resolve().is_relative_to(args.dataset.resolve()):
        raise ValueError("A token report cannot modify the frozen dataset")
    manifest, system_file = args.dataset / "mixed.jsonl", args.dataset / "system_prompt.txt"
    records = read_records(manifest, training=True)
    if any(any(row.get(key) for key in ("image", "images", "image_path", "video", "videos")) for row in records):
        raise ValueError("This CPU audit accepts text-only training records; image tokens require another audit")
    system_prompt = system_file.read_text(encoding="utf-8-sig").strip()
    if not system_prompt:
        raise ValueError("Empty system prompt")

    # Mirror the offline training processor exactly; no GPU model is loaded.
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
    sys.addaudithook(_deny_network)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(eos_id, int) or eos_id == processor.tokenizer.unk_token_id:
        raise ValueError("Cannot identify Qwen assistant EOS")

    def encode(row: dict, *, target: bool):
        question = row["conversations"][0]["value"].replace("<image>", "", 1).strip()
        turns = [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": [{"type": "text", "text": question}]}]
        if target:
            turns.append({"role": "assistant", "content": [{"type": "text", "text": row["conversations"][1]["value"]}]})
        rendered = processor.apply_chat_template(turns, tokenize=False, add_generation_prompt=not target)
        return processor(text=[rendered], return_tensors="pt")

    measured = []
    for row in records:
        full = encode(row, target=True)
        prompt = encode(row, target=False)
        prefix_size = prompt["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :prefix_size] = -100
        labels[full["attention_mask"] == 0] = -100
        ids = full["input_ids"][0].tolist()
        prompt_ids = prompt["input_ids"][0].tolist()
        label_ids = labels[0].tolist()
        # Collect oversize rows without truncation, but still apply all other
        # exact trainer checks to their original unmodified sequences.
        stats = validate_training_tokens(ids, prompt_ids, label_ids,
                                         max_sequence_tokens=max(args.max_sequence_tokens, len(ids)),
                                         identity=record_id(row))
        supervised_eos = any(token == eos_id and label == eos_id for token, label in zip(ids, label_ids))
        if not supervised_eos:
            raise ValueError("Assistant EOS is not supervised: " + record_id(row))
        measured.append({"record_id": record_id(row), "sequence_tokens": stats["sequence_tokens"],
                         "prompt_tokens": stats["prompt_tokens"], "supervised_tokens": stats["supervised_tokens"],
                         "assistant_eos_supervised": supervised_eos})
    longest = sorted(measured, key=lambda item: (-item["sequence_tokens"], item["record_id"]))
    oversize = [item for item in longest if item["sequence_tokens"] > args.max_sequence_tokens]
    report = {"schema": "qwen_facies_cpu_token_audit_v1", "rows": len(measured), "device": "CPU only",
              "network_disabled": True, "truncation_applied": False,
              "manifest_sha256": digest(manifest), "system_prompt_sha256": digest(system_file),
              "base_config_sha256": digest(args.model / "config.json"), "base_snapshot": args.model.name,
              "max_sequence_tokens": args.max_sequence_tokens, "prior_observed_tokens": args.prior_observed_tokens,
              "max_observed_tokens": longest[0]["sequence_tokens"],
              "ratio_to_previous_max": round(longest[0]["sequence_tokens"] / args.prior_observed_tokens, 4),
              "over_previous_max_rows": sum(item["sequence_tokens"] > args.prior_observed_tokens for item in measured),
              "over_limit_rows": len(oversize), "longest_five": longest[:5],
              "limitations": ["717 tokens is a previous observed maximum, not a guaranteed 8 GB memory limit.",
                              "Input length and source audit do not establish geological accuracy or GPU memory fit."]}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    if oversize:
        raise SystemExit("Some full training examples exceed the configured maximum; none were truncated.")


if __name__ == "__main__":
    main()
