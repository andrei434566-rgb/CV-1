"""Experimental single-GPU LoRA fine-tuning for the core-description VLM.

This script intentionally refuses unreviewed rows unless
--experimental-candidates is supplied. It does not infer depositional labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    AutoProcessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "qwen3_vl_candidates.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def read_records(manifest: Path, include_candidates: bool) -> list[dict]:
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not include_candidates:
        records = [r for r in records if r.get("metadata", {}).get("training_approved") is True]
    if not records:
        raise ValueError(
            "No approved records found. Review the candidate pairs first, or pass "
            "--experimental-candidates for an explicitly experimental run."
        )
    return records


def read_manifests(manifests: list[Path], include_candidates: bool) -> list[dict]:
    records = []
    for manifest in manifests:
        records.extend(read_records(manifest, include_candidates))
    return records


def provenance_counts(records: list[dict]) -> dict:
    source_groups: set[str] = set()
    well_ids: set[str] = set()
    all_records_have_well_id = True
    for record in records:
        metadata = record.get("metadata", {})
        source = metadata.get("dataset") or metadata.get("source_group") or metadata.get("source_well")
        row_sources = metadata.get("sources") or []
        if row_sources:
            for item in row_sources:
                source_groups.add(str(item.get("source_id") or item))
        image_name = Path(record.get("image", "")).name.lower()
        if image_name.startswith("urengoy_41_"):
            well_id = "R-41"
            source_groups.add("well_R-41")
        else:
            well_id = metadata.get("well_id") or metadata.get("source_well")
            if not row_sources:
                source_groups.add(str(source or "unknown_source"))
        if well_id:
            well_ids.add(str(well_id))
        else:
            all_records_have_well_id = False
    record_type_counts = {}
    for record in records:
        record_type = record.get("metadata", {}).get("record_type") or (
            "image_text" if record.get("image") else "text_only"
        )
        record_type_counts[record_type] = record_type_counts.get(record_type, 0) + 1
    return {
        "independent_data_sources": len(source_groups),
        "source_groups": sorted(source_groups),
        "independent_wells": len(well_ids) if all_records_have_well_id else None,
        "record_type_counts": record_type_counts,
        "independent_well_count_note": "Core collection datasets are not assumed to be wells; count is null unless every row identifies a well.",
    }


class CorePhotoDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], processor, max_image_side: int):
        self.records = records
        self.processor = processor
        self.max_image_side = max_image_side

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = None
        if record.get("image"):
            image_path = Path(record["image"])
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                image.thumbnail((self.max_image_side, self.max_image_side))

        human_text = record["conversations"][0]["value"].replace("<image>", "", 1).strip()
        target_text = record["conversations"][1]["value"]
        if image is None:
            user_content = [{"type": "text", "text": human_text}]
        else:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": human_text},
            ]
        user_message = {"role": "user", "content": user_content}
        full_messages = [
            user_message,
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )

        if image is None:
            full = self.processor(text=[full_text], return_tensors="pt")
            prompt = self.processor(text=[prompt_text], return_tensors="pt")
        else:
            full = self.processor(text=[full_text], images=[image], return_tensors="pt")
            prompt = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        labels = full["input_ids"].clone()
        prompt_length = prompt["input_ids"].shape[1]
        labels[:, :prompt_length] = -100
        if "attention_mask" in full:
            labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        return {key: value.squeeze(0) if key in {"input_ids", "attention_mask", "labels"} else value
                for key, value in full.items()}


class SingleImageCollator:
    """Pad text and concatenate variable visual patches for a one-image batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(item["input_ids"].shape[-1] for item in features)
        batch = len(features)
        input_ids = torch.full((batch, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        labels = torch.full((batch, max_len), -100, dtype=torch.long)
        for i, item in enumerate(features):
            size = item["input_ids"].shape[-1]
            input_ids[i, :size] = item["input_ids"]
            attention_mask[i, :size] = item["attention_mask"]
            labels[i, :size] = item["labels"]

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        visual_keys = set.intersection(*(set(x.keys()) for x in features)) - set(result)
        for key in visual_keys:
            tensors = [x[key] for x in features]
            result[key] = torch.cat(tensors, dim=0) if tensors[0].ndim else torch.stack(tensors)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--additional-manifest",
        type=Path,
        action="append",
        default=[],
        help="Additional approved JSONL manifest(s), including text-only examples.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "core_vlm_v1_lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Continue training an existing LoRA adapter instead of creating a fresh one.",
    )
    parser.add_argument("--experimental-candidates", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required; the detected build is CPU-only.")
    records = read_manifests(
        [args.manifest, *args.additional_manifest], args.experimental_candidates
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if args.initial_adapter:
        model = PeftModel.from_pretrained(
            model,
            args.initial_adapter,
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = CorePhotoDataset(records, processor, args.max_image_side)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SingleImageCollator(processor.tokenizer.pad_token_id),
        processing_class=processor,
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)
    metrics = train_result.metrics
    metrics.update(
        {
            "candidate_rows_used": len(records),
            **provenance_counts(records),
            "training_approved_rows_used": sum(
                r.get("metadata", {}).get("training_approved") is True for r in records
            ),
            "geology_reviewed_rows_used": sum(
                r.get("metadata", {}).get("geology_approved") is True for r in records
            ),
            "validation_metrics": None,
            "training_status": (
                "source_checked_text_theory_not_geologist_reviewed"
                if all(
                    r.get("metadata", {}).get("training_approved") is True
                    and r.get("metadata", {}).get("record_type") == "text_only_knowledge"
                    and str(r.get("metadata", {}).get("review_status", "")).startswith("source_checked")
                    for r in records
                )
                else
                "reviewed_geology_and_user_approved_pilot"
                if all(
                    r.get("metadata", {}).get("training_approved") is True
                    and r.get("metadata", {}).get("geology_approved") is True
                    for r in records
                )
                else "source_labeled_geology_not_reviewed"
                if all(
                    r.get("metadata", {}).get("training_approved") is True
                    and str(r.get("metadata", {}).get("label_provenance", "")).startswith("source_direct")
                    for r in records
                )
                else "user_approved_pairing_geology_not_reviewed"
                if all(r.get("metadata", {}).get("training_approved") is True for r in records)
                else "experimental_unreviewed_candidates"
            ),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
