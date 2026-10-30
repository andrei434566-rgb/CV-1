"""A bounded local continuation, with a frozen before/after development set.

Image targets are original Mendeley rock labels. Facies examples are explicitly
text-only source-checked hypothetical questions, never photo facies labels.
No model is activated by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = ("песчаник", "известняк", "глинистая порода", "не керн")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def restore_recipe(args, saved=None):
    for name, key, default in (("count", "image_rows", 800), ("learning_rate", "learning_rate", 5e-6),
                               ("seed", "seed", 20261030)):
        supplied = getattr(args, name)
        expected = saved[key] if saved is not None else default
        if saved is not None and supplied is not None and supplied != expected:
            raise ValueError("Resume cannot change the saved training recipe: " + name)
        setattr(args, name, expected if supplied is None or saved is not None else supplied)


def complete_checkpoint(root):
    """Skip interrupted writes; inspect archives without loading pickle code."""
    from safetensors import safe_open, SafetensorError
    candidates = sorted((p for p in root.glob("checkpoint-*") if p.name[11:].isdigit()),
                        key=lambda p: int(p.name[11:]), reverse=True)
    required = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth")
    for folder in candidates:
        try:
            if not all((folder / name).is_file() and (folder / name).stat().st_size for name in required):
                continue
            state = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
            if state["global_step"] != int(folder.name[11:]):
                continue
            with safe_open(str(folder / "adapter_model.safetensors"), framework="pt", device="cpu") as weights:
                if not list(weights.keys()):
                    continue
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"):
                path = folder / name
                if path.exists():
                    with zipfile.ZipFile(path) as archive:
                        if archive.testzip() is not None:
                            raise ValueError("Incomplete training state archive")
            return folder
        except (OSError, ValueError, KeyError, SafetensorError, zipfile.BadZipFile):
            continue
    raise ValueError("No complete saved checkpoint to resume")


def reusable_prior(folder, config, image_rows, text_rows):
    previous = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    for key in ("base", "base_revision", "initial_adapter_sha256", "seed", "train_manifest_sha256", "validation_manifest_sha256"):
        if previous.get(key) != config.get(key):
            raise ValueError("Before-evaluation reuse mismatch: " + key)
    snapshot = folder / "training_script_snapshot.py"
    if digest(snapshot) != previous["script_sha256"]:
        raise ValueError("Before-evaluation script checksum mismatch")
    # Check the exact inference implementation, independently of training CLI
    # changes such as Transformers' removal of warmup_ratio.
    import ast
    def inference_ast(path):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        return ast.dump(next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "evaluate"))
    if inference_ast(snapshot) != inference_ast(__file__):
        raise ValueError("Before-evaluation generation code differs")
    answers = json.loads((folder / "prior_adapter_answers.json").read_text(encoding="utf-8"))
    expected = image_rows + text_rows
    if len(answers) != len(expected) or len({row['record_id'] for row in answers}) != len(answers):
        raise ValueError("Incomplete before-evaluation answers")
    for answer, row in zip(answers, expected):
        if answer["record_id"] != row["metadata"]["record_id"]:
            raise ValueError("Before-evaluation identity differs")
        if row.get("image"):
            if answer["expected"] != row["conversations"][1]["value"] or answer["well_id"] != row["metadata"]["well_id"]:
                raise ValueError("Before-evaluation source class differs")
        elif (answer["question"] != row["conversations"][0]["value"] or answer["reference"] != row["conversations"][1]["value"]):
            raise ValueError("Before-evaluation question or reference differs")
    return answers


def balanced(rows, count, seed):
    if count <= 0 or count % len(LABELS):
        raise ValueError("Sample count must be a positive multiple of four")
    rng, result = random.Random(seed), []
    for label in LABELS:
        group = sorted((row for row in rows if row["conversations"][1]["value"] == label),
                       key=lambda row: row["metadata"]["record_id"])
        if len(group) < count // 4:
            raise ValueError("Insufficient original source labels for " + label)
        result.extend(rng.sample(group, count // 4))
    rng.shuffle(result)
    return result


def validate_split(train, validation, previous):
    train_hashes = {row["metadata"]["image_sha256"] for row in train}
    val_hashes = {row["metadata"]["image_sha256"] for row in validation}
    previous_hashes = {row["metadata"]["image_sha256"] for row in previous}
    if len(train_hashes) != len(train) or train_hashes & (val_hashes | previous_hashes):
        raise ValueError("Duplicate image, validation leakage, or repeated continuation image")
    wells = lambda rows: {row["metadata"]["well_id"] for row in rows}
    if wells(train + previous) & wells(validation):
        raise ValueError("Training/validation wells overlap")
    for row in train + validation:
        meta = row["metadata"]
        if meta.get("label_provenance") != "source_direct_folder":
            raise ValueError("Only original publisher lithology classes are accepted")
        if row in train and (meta.get("training_approved") is not True or meta.get("well_split") != "train"):
            raise ValueError("Continuation requires an approved training row")
        if row in validation and meta.get("well_split") != "val":
            raise ValueError("The sealed test split is not used")
        if digest(row["image"]) != meta["image_sha256"]:
            raise ValueError("Image checksum differs from source manifest")


def prepare(runtime, output, count, seed):
    previous = read_rows(runtime / "data/mendeley_lithology_4000/train.jsonl")
    train = balanced(read_rows(runtime / "data/mendeley_lithology_next/train.jsonl"), count, seed)
    validation = balanced(read_rows(runtime / "data/mendeley_lithology_expanded/val.jsonl"), 64, seed)
    validate_split(train, validation, previous)
    theory = []
    for name in ("training_reasoned_cards_v3.json", "training_positive_extension_v3.json"):
        document = json.loads((ROOT / "vlm_source/facies_reasoning_v1" / name).read_text(encoding="utf-8"))
        for card in document["cards"]:
            if card.get("source_checked") is not True or card.get("split") != "train" or not card.get("source_ids"):
                raise ValueError("Unverified facies theory card")
            theory.append({"conversations": [{"from": "human", "value": card["question"]},
                                               {"from": "gpt", "value": card["answer"]}],
                           "metadata": {"record_id": card["id"], "record_type": "text_only_facies_reasoning",
                                        "training_approved": True, "geology_approved": False,
                                        "source_checked": True, "source_ids": card["source_ids"], "split": "train"}})
    evaluation = json.loads((ROOT / "vlm_source/facies_reasoning_v1/evaluation.json").read_text(encoding="utf-8"))
    selected_ids = {"FRV1-01", "FRV1-02", "FRV1-03", "FRV1-05", "FRV1-07", "FRV1-10"}
    text_validation = [{"conversations": [{"from": "human", "value": item["question"]},
                                           {"from": "gpt", "value": item["answer"]}],
                        "metadata": {"record_id": item["id"], "kind": "facies_reasoning",
                                     "required_facts": item["required_facts"], "sources": item["sources"]}}
                       for item in evaluation["items"] if item["id"] in selected_ids]
    for number, question in enumerate(("Привет! Чем ты можешь помочь?", "Можно ли по фото точно определить фацию?")):
        text_validation.append({"conversations": [{"from": "human", "value": question},
                                                    {"from": "gpt", "value": "qualitative conversation guard"}],
                                "metadata": {"record_id": f"dialogue_guard_{number}", "kind": "conversation"}})
    for name, rows in (("train", train + theory), ("validation_images", validation), ("validation_text", text_validation)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return train + theory, validation, text_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT.parent / "vlm_core_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prior-evaluation", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume the last checkpoint in the same output directory")
    args = parser.parse_args()
    if args.resume and (args.output / "result.json").is_file():
        raise ValueError("This run is complete; start a new output directory for another experiment")
    if args.resume and not args.output.is_dir():
        raise ValueError("Resume requires an existing run directory")
    if args.output.exists() and not args.resume:
        raise ValueError("Output already exists; use a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    saved_config = json.loads((args.output / "run_config.json").read_text(encoding="utf-8")) if args.resume else None
    restore_recipe(args, saved_config)
    checkpoint = complete_checkpoint(args.output / "trainer") if args.resume else None
    if args.resume:
        train = read_rows(args.output / "train.jsonl")
        images, texts = read_rows(args.output / "validation_images.jsonl"), read_rows(args.output / "validation_text.jsonl")
        validate_split([row for row in train if row.get("image")], images,
                       read_rows(args.runtime / "data/mendeley_lithology_4000/train.jsonl"))
        args.reuse_prior_evaluation = args.output
    else:
        train, images, texts = prepare(args.runtime, args.output, args.count, args.seed)
    base = args.runtime / "model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
    initial = args.runtime / "outputs/core_vlm_mendeley_lithology_4000_lora"
    config = {"base": str(base), "base_revision": base.name, "initial_adapter": str(initial),
              "initial_adapter_sha256": digest(initial / "adapter_model.safetensors"),
              "script_sha256": digest(__file__), "seed": args.seed, "epochs": 1,
              "learning_rate": args.learning_rate, "image_rows": args.count, "text_rows": len(train) - args.count,
              "train_classes": dict(Counter(row["conversations"][1]["value"] for row in train if row.get("image"))),
              "train_manifest_sha256": digest(args.output / "train.jsonl"),
              "validation_manifest_sha256": digest(args.output / "validation_images.jsonl"),
              "source": "https://data.mendeley.com/datasets/b6rj29mrss/1", "license": "CC BY 4.0",
              "status": "prepared_not_activated", "facies_photo_labels": 0,
              "limitation": "Development validation, previously examined wells; not independent final test accuracy"}
    cached_prior = reusable_prior(args.reuse_prior_evaluation, config, images, texts) if args.reuse_prior_evaluation else None
    if cached_prior:
        config["reused_prior_evaluation"] = {"folder": str(args.reuse_prior_evaluation),
            "answers_sha256": digest(args.reuse_prior_evaluation / "prior_adapter_answers.json"),
            "validation": "matching base, adapter hash, seed, data hashes, exact inference AST, source IDs and references"}
    if not args.resume:
        write_json(args.output / "run_config.json", config)
        shutil.copy2(__file__, args.output / "training_script_snapshot.py")
    else:
        write_json(args.output / "resume_config.json", config)
    if args.prepare_only:
        print(json.dumps(config, ensure_ascii=True), flush=True)
        return
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, set_seed
    sys.path.insert(0, str(args.runtime))
    from train_qwen3vl import CorePhotoDataset, SingleImageCollator
    from scripts.train_qwen_source_pilot import sft_supervised_token_loss, validate_training_tokens
    arguments = TrainingArguments(output_dir=str(args.output / "trainer"), num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=args.learning_rate,
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", logging_steps=10, save_strategy="steps", save_steps=50, save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0, report_to="none", seed=args.seed, data_seed=args.seed,
        warmup_steps=math.ceil(math.ceil(len(train) / 4) * .03), disable_tqdm=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    free, total = torch.cuda.mem_get_info()
    if free < 6.5 * 1024**3:
        raise RuntimeError("Недостаточно свободной видеопамяти; сохраните проект и закройте загруженную VLM.")
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    dataset = CorePhotoDataset(train, processor, 300)
    # Independently reconstruct each prompt and verify masking before loading
    # model weights. Never silently truncate an oversize or mismatched target.
    from PIL import Image
    token_summary = {"rows": len(train), "max_sequence_tokens": 0, "truncated_rows": 0}
    for index, row in enumerate(train):
        features = dataset[index]
        question = row["conversations"][0]["value"].replace("<image>", "", 1).strip()
        content = [{"type": "text", "text": question}]
        image = None
        if row.get("image"):
            with Image.open(row["image"]) as opened:
                image = opened.convert("RGB")
                image.thumbnail((300, 300))
            content.insert(0, {"type": "image", "image": image})
        rendered = processor.apply_chat_template([{"role": "user", "content": content}],
                                                 tokenize=False, add_generation_prompt=True)
        options = {"images": [image]} if image is not None else {}
        prompt = processor(text=[rendered], return_tensors="pt", **options)
        counts = validate_training_tokens(features["input_ids"].tolist(), prompt["input_ids"][0].tolist(),
            features["labels"].tolist(), max_sequence_tokens=1024, identity=row["metadata"]["record_id"])
        token_summary["max_sequence_tokens"] = max(token_summary["max_sequence_tokens"], counts["sequence_tokens"])
    write_json(args.output / "token_preflight.json", token_summary)
    model = Qwen3VLForConditionalGeneration.from_pretrained(base, local_files_only=True, dtype=torch.float16,
                                                          attn_implementation="sdpa", device_map="cuda")
    model = PeftModel.from_pretrained(model, initial, is_trainable=True)
    log_path = args.output / "progress.jsonl"
    def log(event, **values):
        data = {"event": event, **values}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(json.dumps(data, ensure_ascii=True), flush=True)
    def evaluate(stage):
        from evaluate_barnegat_binary_pilot import predict
        model.eval()
        model.config.use_cache = True
        answers = []
        for index, row in enumerate(images + texts):
            if row.get("image"):
                answer = predict(model, processor, row)
                expected = row["conversations"][1]["value"]
                predicted = answer.casefold().strip().strip(' .,:;!"«»')
                result = {"kind": "lithology", "correct": predicted == expected, "expected": expected, "prediction": predicted,
                          "well_id": row["metadata"]["well_id"]}
            else:
                prompt = processor.apply_chat_template([{"role": "user", "content": row["conversations"][0]["value"]}],
                                                       tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                new_ids = generated[:, inputs["input_ids"].shape[1]:]
                answer = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
                result = {"kind": row["metadata"]["kind"], "reference": row["conversations"][1]["value"],
                          "question": row["conversations"][0]["value"], "truncated": new_ids.shape[1] >= 200}
                del inputs, generated, new_ids
            answers.append({"record_id": row["metadata"]["record_id"], "answer": answer, **result})
            log("evaluation", stage=stage, index=index + 1, total=len(images + texts), **answers[-1])
            write_json(args.output / f"{stage}_answers.json", answers)
        return answers
    before = cached_prior if cached_prior is not None else evaluate("prior_adapter")
    if cached_prior is not None:
        write_json(args.output / "prior_adapter_answers.json", cached_prior)
        log("prior_evaluation_reused", rows=len(cached_prior))
    model.train()
    model.config.use_cache = False
    class TokenLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = sft_supervised_token_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            return (loss, outputs) if return_outputs else loss
    class Callback(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            log("train", step=state.global_step, **(logs or {}))
    trainer = TokenLossTrainer(model=model, args=arguments, train_dataset=dataset,
                              data_collator=SingleImageCollator(processor.tokenizer.pad_token_id),
                              processing_class=processor, callbacks=[Callback()])
    started = perf_counter()
    log("training_started", rows=len(train))
    if args.resume:
        log("resuming_checkpoint", checkpoint=str(checkpoint))
    training = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    adapter = args.output / "adapter"
    trainer.save_model(str(adapter))
    processor.save_pretrained(adapter)
    model.gradient_checkpointing_disable()
    model.zero_grad(set_to_none=True)
    if trainer.optimizer:
        trainer.optimizer.state.clear()
    torch.cuda.empty_cache()
    after = evaluate("candidate")
    def score(rows):
        selected = [row for row in rows if row["kind"] == "lithology"]
        result = {"correct": sum(row["correct"] for row in selected), "total": len(selected),
                  "predictions": dict(Counter(row["prediction"] for row in selected)), "recall": {}}
        for label in LABELS:
            subset = [row for row in selected if row["expected"] == label]
            result["recall"][label] = sum(row["correct"] for row in subset) / len(subset)
        return result
    result = {"prior_adapter": score(before), "candidate": score(after), "training": training.metrics,
              "seconds_including_after_evaluation": perf_counter() - started,
              "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
              "status": "candidate_requires_review_not_installed", "facies_accuracy": None}
    write_json(args.output / "result.json", result)
    log("complete", **result)


if __name__ == "__main__":
    main()
