"""Offline Qwen3-VL source-label pilot: base comparison, fresh LoRA, comparison.

Only approved training rows are accepted. Validation is never used for loss or
checkpoint selection; the sealed test split is refused. Saved adapters are
candidates and are never installed into Kern Analyzer by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

def configure_offline_environment() -> None:
    """Configure the CLI before ML imports, without changing helper importers."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ.update(WANDB_DISABLED="true", HF_HUB_DISABLE_TELEMETRY="1",
                      DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false")


def _deny_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
        raise PermissionError("Network access is disabled for local Qwen training.")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_id(record: dict) -> str:
    return str(record.get("id") or record.get("metadata", {}).get("record_id")
               or hashlib.sha256(json.dumps(record["conversations"], ensure_ascii=False,
                                            sort_keys=True).encode("utf-8")).hexdigest()[:20])


def read_records(path: Path, *, training: bool) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata", {})
        split = str(meta.get("split") or meta.get("partition") or row.get("split", "")).lower()
        if split == "test":
            raise ValueError(f"Sealed test row refused: {path.name}:{line_number}")
        if training and (meta.get("training_approved") is not True or split != "train"):
            raise ValueError(f"Only explicitly approved train rows are accepted: {path.name}:{line_number}")
        if not training and split not in {"val", "validation", "holdout", ""}:
            raise ValueError(f"Unexpected validation split {split!r}: {path.name}:{line_number}")
        turns = row.get("conversations", [])
        if len(turns) != 2 or not all(str(turn.get("value", "")).strip() for turn in turns):
            raise ValueError(f"Expected one complete question/answer pair: {path.name}:{line_number}")
        if row.get("image"):
            image = Path(row["image"])
            if not image.is_absolute():
                image = path.parent / image
            if not image.is_file():
                raise FileNotFoundError(image)
            row["image"] = str(image.resolve())
            if training and not str(meta.get("label_provenance", "")).startswith("source_direct"):
                raise ValueError(f"Image row lacks direct source-label provenance: {record_id(row)}")
        records.append(row)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    ids = [record_id(row) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate record IDs in {path}")
    return records


def normalized_label(value: str) -> str:
    value = value.casefold().replace("ё", "е").replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", value).strip(" \r\n\t.,;:!?«»\"'")


def source_id(row: dict) -> str:
    meta = row.get("metadata", {})
    return str(meta.get("source_id") or meta.get("dataset") or meta.get("source_group") or "unknown_source")


def target_label(row: dict) -> str:
    meta = row.get("metadata", {})
    return str(meta.get("target_label_ru") or meta.get("target_label") or row["conversations"][1]["value"]).strip()


def aliases(row: dict) -> set[str]:
    values = row.get("metadata", {}).get("label_aliases", [])
    if isinstance(values, str):
        values = [values]
    return {normalized_label(str(value)) for value in [target_label(row), *values]}


def classification_metrics(rows: list[dict], responses: list[dict]) -> dict:
    answers = {row["record_id"]: row for row in responses if row["kind"] == "image_classification"}
    result = {}
    for source in sorted({source_id(row) for row in rows}):
        selected = [row for row in rows if source_id(row) == source]
        labels = sorted({target_label(row) for row in selected})
        label_aliases = {label: set() for label in labels}
        for row in selected:
            label_aliases[target_label(row)].update(aliases(row))
        confusion = {label: {pred: 0 for pred in [*labels, "__unknown__"]} for label in labels}
        truncated = 0
        for row in selected:
            answer = answers[record_id(row)]
            text = normalized_label(answer["response"])
            matches = [label for label, names in label_aliases.items() if text in names]
            predicted = matches[0] if len(matches) == 1 else "__unknown__"
            confusion[target_label(row)][predicted] += 1
            truncated += bool(answer["truncated"])
        correct = sum(confusion[label][label] for label in labels)
        f1_scores = []
        for label in labels:
            tp = confusion[label][label]
            fp = sum(confusion[other][label] for other in labels if other != label)
            fn = sum(confusion[label].values()) - tp
            f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        result[source] = {
            "rows": len(selected), "correct": correct, "accuracy": correct / len(selected),
            "macro_f1": sum(f1_scores) / len(f1_scores), "confusion_matrix": confusion,
            "unknown_output": sum(confusion[label]["__unknown__"] for label in labels),
            "truncated": truncated,
        }
    return result


def evaluation_kind(row: dict) -> str:
    if row.get("image"):
        return "image_classification"
    return "casual_guard" if row.get("metadata", {}).get("evaluation_kind") == "casual_guard" else "theory"


def evaluation_metrics(image_rows: list[dict], eval_rows: list[dict], responses: list[dict], seconds: float) -> dict:
    theory = [row for row in responses if row["kind"] == "theory"]
    casual = [row for row in responses if row["kind"] == "casual_guard"]
    return {
        "image_classification": classification_metrics(image_rows, responses),
        "theory": {"rows": len(theory), "accuracy": None,
                   "evaluation": "qualitative; raw responses and references saved",
                   "truncated": sum(row["truncated"] for row in theory)},
        "casual_guard": {"rows": len(casual), "accuracy": None,
                         "evaluation": "qualitative conversation regression guard",
                         "empty_answers": sum(not row["response"] for row in casual)},
        "seconds": round(seconds, 3),
    }


def load_reusable_baseline(folder: Path, settings: dict, eval_rows: list[dict]) -> tuple[list[dict], dict]:
    """Validate the full comparison identity before reusing any baseline answers."""
    config_path = folder / "run_config.json"
    responses_path = folder / "baseline_responses.jsonl"
    previous = json.loads(config_path.read_text(encoding="utf-8-sig"))
    fields = ("schema", "model", "model_config_sha256", "max_image_side", "seed", "generation",
              "theory_evaluated_ids", "image_validation_rows", "torch", "transformers", "peft", "cuda")
    for field in fields:
        if previous.get(field) != settings.get(field):
            raise ValueError(f"Baseline reuse rejected: {field} differs")
    old_manifests = previous.get("manifests", {})
    new_manifests = settings.get("manifests", {})
    if set(old_manifests) != set(new_manifests):
        raise ValueError("Baseline reuse rejected: manifest set differs")
    for name, value in new_manifests.items():
        if old_manifests[name].get("sha256") != value.get("sha256"):
            raise ValueError(f"Baseline reuse rejected: {name} manifest checksum differs")
    responses = [json.loads(line) for line in responses_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    expected = {record_id(row): row for row in eval_rows}
    if len(expected) != len(eval_rows) or len(responses) != len(eval_rows):
        raise ValueError("Baseline reuse rejected: incomplete or duplicate expected rows")
    observed = set()
    for answer in responses:
        identity = answer.get("record_id")
        if identity not in expected or identity in observed:
            raise ValueError("Baseline reuse rejected: unexpected or duplicate response ID")
        observed.add(identity)
        row = expected[identity]
        if (answer.get("stage") != "baseline" or answer.get("kind") != evaluation_kind(row)
                or answer.get("source_id") != source_id(row)
                or answer.get("question") != row["conversations"][0]["value"]
                or answer.get("reference") != row["conversations"][1]["value"]):
            raise ValueError(f"Baseline reuse rejected: response identity differs for {identity}")
        if (not isinstance(answer.get("response"), str) or type(answer.get("truncated")) is not bool
                or type(answer.get("generated_tokens")) is not int or answer["generated_tokens"] < 0
                or not isinstance(answer.get("seconds"), (int, float))
                or not math.isfinite(answer["seconds"]) or answer["seconds"] < 0):
            raise ValueError(f"Baseline reuse rejected: malformed response for {identity}")
    if observed != set(expected):
        raise ValueError("Baseline reuse rejected: missing response IDs")
    provenance = {
        "source_directory": str(folder.resolve()), "run_config_sha256": sha256(config_path),
        "baseline_responses_sha256": sha256(responses_path), "rows": len(responses),
        "validation": "exact model/config, manifest hashes, generation, image size, seed, versions, IDs, kinds, questions and references",
        "metrics_recomputed_from_raw_responses": True,
    }
    return responses, provenance


def training_arguments_kwargs(output_dir: Path, *, epochs: float, learning_rate: float, rows: int, seed: int) -> dict:
    total_steps = math.ceil(math.ceil(rows / 4) * epochs)
    return {
        "output_dir": str(output_dir / "trainer"), "num_train_epochs": epochs,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate, "fp16": True, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}, "optim": "adamw_torch",
        "logging_steps": 10, "save_strategy": "no", "remove_unused_columns": False,
        "dataloader_num_workers": 0, "report_to": "none", "seed": seed, "data_seed": seed,
        "warmup_steps": math.ceil(total_steps * 0.03), "lr_scheduler_type": "linear", "disable_tqdm": True,
    }


def choose_theory(rows: list[dict], limit: int) -> list[dict]:
    """Deterministic topic round robin; same recorded IDs for base and adapter."""
    groups = {}
    for row in rows:
        if row.get("image"):
            raise ValueError("Text validation manifest contains images")
        groups.setdefault(str(row.get("metadata", {}).get("topic", "general")), []).append(row)
    rng = random.Random(42)
    for values in groups.values():
        rng.shuffle(values)
    chosen = []
    while groups and len(chosen) < limit:
        for topic in sorted(list(groups)):
            chosen.append(groups[topic].pop(0))
            if not groups[topic]:
                del groups[topic]
            if len(chosen) == limit:
                break
    return chosen


def casual_rows() -> list[dict]:
    questions = ["Работаешь?", "Чем ты можешь помочь?", "Привет! Давай познакомимся.",
                 "Спасибо. Я сейчас задам другой вопрос, без фотографии."]
    return [{"conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": ""}],
             "metadata": {"record_id": f"casual_guard_{index + 1}", "record_type": "casual_guard"}}
            for index, question in enumerate(questions)]


class Progress:
    def __init__(self, folder: Path):
        self.stream = (folder / "progress.jsonl").open("a", encoding="utf-8", buffering=1)

    def __call__(self, event: str, **values):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def main() -> None:
    # This CLI is also imported by CPU-only checks; isolate the network rule to runs.
    configure_offline_environment()
    sys.addaudithook(_deny_network)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--val-text-manifest", required=True, type=Path)
    parser.add_argument("--val-conversation-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--system-prompt-file", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-image-side", type=int, default=280)
    parser.add_argument("--theory-limit", type=int, default=8)
    parser.add_argument("--theory-max-new-tokens", type=int, default=260)
    parser.add_argument("--classification-max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--reuse-baseline-dir", type=Path,
                        help="Reuse a complete baseline only when inputs, settings and every response identity match exactly")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output_dir}")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError("A complete local Qwen3-VL model folder is required")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.max_image_side < 56:
        raise ValueError("Invalid training parameters")
    train_rows = read_records(args.manifest, training=True)
    image_eval = read_records(args.eval_manifest, training=False)
    if not all(row.get("image") for row in image_eval):
        raise ValueError("Image validation manifest must contain only image classification rows")
    all_theory = read_records(args.val_text_manifest, training=False)
    theory_eval = choose_theory(all_theory, args.theory_limit)
    conversation_eval = (read_records(args.val_conversation_manifest, training=False)
                         if args.val_conversation_manifest else casual_rows())
    if any(row.get("image") for row in conversation_eval):
        raise ValueError("Conversation regression manifest must contain only text")
    for row in conversation_eval:
        row["metadata"]["evaluation_kind"] = "casual_guard"
    eval_rows = [*image_eval, *theory_eval, *conversation_eval]
    overlap = {record_id(row) for row in train_rows} & {record_id(row) for row in eval_rows}
    if overlap:
        raise ValueError(f"Training/validation record ID overlap: {sorted(overlap)}")
    train_questions = {normalized_label(row["conversations"][0]["value"]) for row in train_rows if not row.get("image")}
    if any(normalized_label(row["conversations"][0]["value"]) in train_questions for row in theory_eval):
        raise ValueError("Theory validation question is also present in training")
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8-sig").strip()
    image_keys = lambda rows: {row.get("metadata", {}).get("prepared_pixel_sha256") or row.get("image")
                               for row in rows if row.get("image")}
    if image_keys(train_rows) & image_keys(image_eval):
        raise ValueError("Training/validation prepared image overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.output_dir)
    settings = None
    try:
        import torch
        import transformers
        import peft
        from PIL import Image
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments, set_seed
        from app.infrastructure.ml.qwen_train import CorePhotoDataset, SingleImageCollator

        if not torch.cuda.is_available():
            raise RuntimeError("This pilot requires a local CUDA GPU")
        set_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        # Validate the installed Transformers API before loading weights or generating.
        train_kwargs = training_arguments_kwargs(args.output_dir, epochs=args.epochs,
                                                learning_rate=args.learning_rate, rows=len(train_rows), seed=args.seed)
        training_args = TrainingArguments(**train_kwargs)
        progress("training_arguments_preflight", warmup_steps=train_kwargs["warmup_steps"],
                 estimated_optimizer_steps=math.ceil(math.ceil(len(train_rows) / 4) * args.epochs))
        manifest_info = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {
            "training": args.manifest, "image_validation": args.eval_manifest,
            "theory_validation": args.val_text_manifest, "system_prompt": args.system_prompt_file}.items()}
        if args.val_conversation_manifest:
            manifest_info["conversation_validation"] = {
                "path": str(args.val_conversation_manifest.resolve()), "sha256": sha256(args.val_conversation_manifest)}
        settings = {
            "schema": "qwen_source_pilot_v1", "status": "running", "network_disabled": True,
            "model": str(args.model.resolve()), "model_config_sha256": sha256(args.model / "config.json"),
            "manifests": manifest_info, "train_rows": len(train_rows), "image_validation_rows": len(image_eval),
            "theory_validation_rows_total": len(all_theory), "theory_evaluated_ids": [record_id(row) for row in theory_eval],
            "theory_selection": "seed42 topic round robin, identical before and after; qualitative inspection only",
            "seed": args.seed, "epochs": args.epochs, "learning_rate": args.learning_rate,
            "max_image_side": args.max_image_side, "batch_size": 1, "gradient_accumulation": 4,
            "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
            "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "generation": {"do_sample": False, "repetition_penalty": 1.05,
                           "classification_max_new_tokens": args.classification_max_new_tokens,
                           "theory_max_new_tokens": args.theory_max_new_tokens},
            "test_split_used": False, "installed_in_application": False,
        }
        (args.output_dir / "run_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cached_baseline = None
        if args.reuse_baseline_dir:
            cached_baseline, reuse = load_reusable_baseline(args.reuse_baseline_dir, settings, eval_rows)
            settings["baseline_reuse"] = reuse
            (args.output_dir / "baseline_reuse.json").write_text(json.dumps(reuse, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            shutil.copyfile(args.reuse_baseline_dir / "baseline_responses.jsonl", args.output_dir / "baseline_responses.jsonl")
            (args.output_dir / "run_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            progress("baseline_reuse_validated", **reuse)
        progress("loading_model", model=str(args.model), train_rows=len(train_rows), validation_rows=len(eval_rows))
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model), dtype=torch.float16, attn_implementation="sdpa", local_files_only=True,
        ).to("cuda")

        def messages(row, *, target=False):
            question = row["conversations"][0]["value"].replace("<image>", "", 1).strip()
            content = []
            image = None
            if row.get("image"):
                with Image.open(row["image"]) as opened:
                    image = opened.convert("RGB")
                image.thumbnail((args.max_image_side, args.max_image_side))
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": question})
            turns = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
            if target:
                turns.append({"role": "assistant", "content": [{"type": "text", "text": row["conversations"][1]["value"]}]})
            return turns, image

        def encode(row, *, target=False):
            turns, image = messages(row, target=target)
            text = processor.apply_chat_template(turns, tokenize=False, add_generation_prompt=not target)
            return processor(text=[text], **({"images": [image]} if image else {}), return_tensors="pt")

        class CheckedDataset(CorePhotoDataset):
            def __getitem__(self, index):
                row = self.records[index]
                full = encode(row, target=True)
                prompt = encode(row)
                prefix_size = prompt["input_ids"].shape[1]
                if not torch.equal(full["input_ids"][:, :prefix_size], prompt["input_ids"]):
                    raise ValueError(f"Chat template token prefix mismatch: {record_id(row)}")
                labels = full["input_ids"].clone()
                labels[:, :prefix_size] = -100
                labels[full["attention_mask"] == 0] = -100
                if not torch.any(labels != -100):
                    raise ValueError(f"No supervised assistant tokens: {record_id(row)}")
                full["labels"] = labels
                return {key: value.squeeze(0) if key in {"input_ids", "attention_mask", "labels"} else value
                        for key, value in full.items()}

        dataset = CheckedDataset(train_rows, processor, args.max_image_side, system_prompt)
        # Preflight each modality, including explicit terminal token supervision.
        for kind in (False, True):
            index = next((i for i, row in enumerate(train_rows) if bool(row.get("image")) == kind), None)
            if index is not None:
                item = dataset[index]
                progress("training_preflight", modality="image" if kind else "text", record_id=record_id(train_rows[index]),
                         supervised_tokens=int((item["labels"] != -100).sum()),
                         final_token_supervised=bool(item["labels"][-1] != -100))

        def evaluate(stage):
            model.eval()
            model.config.use_cache = True
            responses = []
            path = args.output_dir / f"{stage}_responses.jsonl"
            started = time.perf_counter()
            with path.open("w", encoding="utf-8", buffering=1) as stream:
                for index, row in enumerate(eval_rows):
                    kind = evaluation_kind(row)
                    limit = args.classification_max_new_tokens if kind == "image_classification" else args.theory_max_new_tokens
                    inputs = encode(row).to("cuda")
                    item_started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                                   repetition_penalty=1.05, use_cache=True,
                                                   pad_token_id=processor.tokenizer.pad_token_id)
                    new_ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else (eos or [])
                    answer = {
                        "stage": stage, "record_id": record_id(row), "kind": kind, "source_id": source_id(row),
                        "question": row["conversations"][0]["value"], "reference": row["conversations"][1]["value"],
                        "response": processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                        "generated_tokens": len(new_ids), "truncated": len(new_ids) >= limit and (not new_ids or new_ids[-1] not in eos),
                        "seconds": round(time.perf_counter() - item_started, 3),
                    }
                    responses.append(answer)
                    stream.write(json.dumps(answer, ensure_ascii=False) + "\n")
                    stream.flush()
                    progress("evaluation_row", stage=stage, index=index + 1, total=len(eval_rows), record_id=record_id(row),
                             kind=kind, response=answer["response"], truncated=answer["truncated"], seconds=answer["seconds"])
                    del inputs, generated
            metrics = evaluation_metrics(image_eval, eval_rows, responses, time.perf_counter() - started)
            (args.output_dir / f"{stage}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            progress("evaluation_finished", stage=stage, metrics=metrics)
            return metrics

        if cached_baseline is None:
            baseline_metrics = evaluate("baseline")
        else:
            baseline_metrics = evaluation_metrics(image_eval, eval_rows, cached_baseline,
                                                  sum(row["seconds"] for row in cached_baseline))
            baseline_metrics["reuse_provenance"] = settings["baseline_reuse"]
            (args.output_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            progress("evaluation_finished", stage="baseline", reused=True, metrics=baseline_metrics)
        targets = [name for name, _module in model.named_modules()
                   if "language_model" in name and name.endswith((".q_proj", ".v_proj"))]
        if not targets:
            raise ValueError("No language-only q_proj/v_proj targets found; refusing implicit visual LoRA")
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                                target_modules=targets, bias="none", task_type="CAUSAL_LM"))
        model.config.use_cache = False
        model.print_trainable_parameters()
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        progress("training_started", trainable_parameters=trainable, target_modules=targets)

        class LogCallback(TrainerCallback):
            def on_log(self, arguments, state, control, logs=None, **kwargs):
                values = {key: value for key, value in (logs or {}).items() if isinstance(value, (int, float, str))}
                progress("training_log", step=state.global_step, **values)

        trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                          data_collator=SingleImageCollator(processor.tokenizer.pad_token_id),
                          processing_class=processor, callbacks=[LogCallback()])
        train_result = trainer.train()
        adapter_dir = args.output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        processor.save_pretrained(str(adapter_dir))
        model.gradient_checkpointing_disable()
        progress("adapter_saved", path=str(adapter_dir), training_metrics=train_result.metrics)
        candidate_metrics = evaluate("candidate")
        metrics = {
            **settings, "status": "candidate_requires_geological_validation", "adapter": str(adapter_dir.resolve()),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "training": train_result.metrics, "baseline": baseline_metrics, "candidate": candidate_metrics,
            "accuracy_change_percentage_points": {
                source: 100 * (candidate_metrics["image_classification"][source]["accuracy"] - values["accuracy"])
                for source, values in baseline_metrics["image_classification"].items()},
            "limitations": ["Source-class validation accuracy is not accuracy of full geological descriptions or facies.",
                            "Validation subsets are small; held-out source test images were not used.",
                            "Text and conversation responses require human review; loss is not accuracy.",
                            "Noncommercial datasets can impose restrictions on resulting use.",
                            "Adapter is separate and has not replaced any active application model."],
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        settings.update(status=metrics["status"], finished_utc=metrics["finished_utc"],
                        adapter=metrics["adapter"], metrics=str((args.output_dir / "metrics.json").resolve()))
        (args.output_dir / "run_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        progress("finished", status=metrics["status"], adapter=str(adapter_dir), metrics=str(args.output_dir / "metrics.json"))
    except BaseException as error:
        failure = {"status": "failed", "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "error": str(error), "traceback": traceback.format_exc()}
        (args.output_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if settings is not None:
            settings.update(status="failed", finished_utc=failure["finished_utc"], error=str(error))
            (args.output_dir / "run_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        progress("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
