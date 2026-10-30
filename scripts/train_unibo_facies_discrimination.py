"""Offline research continuation: discriminate the six source facies.

Preserves the original photo/label splits and general-assistant rejection.
The primary objective is six-way CE at the first assistant token, plus a
small full-vocabulary SFT term to retain answer format. No invented geology,
preference labels from validation, new class mapping, or auto-deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_lithology_facies_continuation import complete_checkpoint

PARENT_SHA = "8b03041cba6e75aa4c814a826a820cfd0a2e8fb5af5d579e4585652f883b63f8"
RECIPE = {"epochs": 3, "learning_rate": 1e-5, "batch_size": 1,
          "gradient_accumulation_steps": 4, "seed": 20261030,
          "max_image_side": 300, "attention": "sdpa", "fp16": True,
          "classification_objective": "CE over A-F at first assistant prediction",
          "facies_format_sft_coefficient": 0.1,
          "replay_objective": "unchanged full supervised-token mean SFT",
          "gradient_normalization": "sample mean, Trainer scales accumulation; model_accepts_loss_kwargs=False",
          "class_weights": "facies N/(6*n_class); raw lithology replay=1",
          "save_steps": 50, "save_total_limit": 2,
          "torch_threads": 4, "interop_threads": 1}
CRITERIA = {"facies_macro_f1_min_gain": 0.05, "facies_accuracy_must_increase": True,
            "must_beat_gray_and_24pct_majority": True,
            "all_six_classes_must_have_nonzero_recall": True,
            "lithology_max_lost_correct": 2,
            "scope": "source-domain correlated development data only; never production acceptance"}


def discrimination_loss(model, inputs, code_ids, task_index):
    """One forward; strict causal alignment; no EOS dilution of class CE."""
    import torch
    import torch.nn.functional as F
    from scripts.train_qwen_source_pilot import sft_supervised_token_loss
    if inputs["labels"].shape[0] != 1:
        raise ValueError("This objective requires batch size one")
    if task_index not in range(-1, 6):
        raise ValueError("Unknown task index")
    # Always take a per-example token mean; Trainer handles accumulation.
    sft, outputs = sft_supervised_token_loss(model, inputs, num_items_in_batch=None)
    if task_index == -1:
        return sft, outputs, None
    ids = tuple(code_ids)
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("Six distinct class token IDs required")
    positions = torch.nonzero(inputs["labels"][0] != -100).flatten()
    if not positions.numel() or positions[0].item() == 0:
        raise ValueError("Missing causal assistant prefix")
    first = int(positions[0])
    if inputs["labels"][0, first].item() != ids[task_index]:
        raise ValueError("Class label differs from the first supervised token")
    logits = outputs.logits[:, 0, list(ids)].float()
    target = torch.tensor([task_index], device=logits.device)
    classification = F.cross_entropy(logits, target)
    return classification + RECIPE["facies_format_sft_coefficient"] * sft, outputs, classification


def pairwise_probes(answers):
    """Auxiliary rankings from the SAME images, never independent samples."""
    result = {}
    for first, second in combinations(source.CODES, 2):
        rows = [r for r in answers if r["expected_code"] in (first, second)]
        predictions = [first if r["probabilities_A_to_F"][first] >= r["probabilities_A_to_F"][second] else second for r in rows]
        result[first + second] = source.classification_metrics([r["expected_code"] for r in rows], predictions, (first, second))
    return {"role": "auxiliary_two_choice_rankings_not_six_way_accuracy",
            "unique_validation_images": len(answers), "pairs": result}


def research_gates(before, after):
    old, new = before["facies"]["forced_choice"], after["facies"]["forced_choice"]
    recalls = new["per_class_recall"]
    return {"macro_f1_gain": new["macro_f1"] >= old["macro_f1"] + CRITERIA["facies_macro_f1_min_gain"],
            "accuracy_gain": new["accuracy"] > old["accuracy"],
            "beats_gray_and_majority": new["accuracy"] > max(after["facies"]["gray_forced_choice"]["accuracy"], .24),
            "all_six_nonzero_recall": set(recalls) == set(source.CODES) and all(v > 0 for v in recalls.values()),
            "lithology_preserved": after["raw_lithology_guard"]["correct"] >= before["raw_lithology_guard"]["correct"] - CRITERIA["lithology_max_lost_correct"]}


def validate_parent_rows(config, parent, runtime):
    for name, key in (("train", "training_manifest_sha256"), ("validation_facies", "facies_validation_manifest_sha256"),
                      ("validation_lithology", "lithology_guard_manifest_sha256")):
        if source.digest(parent / f"{name}.jsonl") != config[key]:
            raise ValueError("Parent manifest changed: " + name)
    for paths in (config["dependency_sha256"], config["source_manifest_sha256"], config["base_config_tokenizer_sha256"]):
        for path, expected in paths.items():
            if source.digest(path) != expected:
                raise ValueError("Parent data or dependency changed: " + path)
    if source.digest(parent / "adapter/adapter_model.safetensors") != PARENT_SHA:
        raise ValueError("Expected the completed visual-facies adapter")
    if config["package_versions"] != source.package_versions() or config["python_version"] != sys.version:
        raise ValueError("Environment differs from the parent experiment")
    data = Path(config["data"])
    raw_train, raw_val = source.read_rows(data / "train.jsonl"), source.read_rows(data / "val.jsonl")
    source.validate_facies_split(raw_train, raw_val)
    rows = source.read_rows(parent / "train.jsonl")
    validation = source.read_rows(parent / "validation_facies.jsonl")
    guard = source.read_rows(parent / "validation_lithology.jsonl")
    actual_facies = {r["metadata"]["record_id"]: r for r in rows if r.get("research_task") == "unibo_six_choice_facies"}
    expected_facies = {r["metadata"]["record_id"]: source.transform_facies(r) for r in raw_train}
    if actual_facies != expected_facies or validation != [source.transform_facies(r) for r in raw_val]:
        raise ValueError("Source labels, prompts, images or approval metadata changed")
    source.validate_lithology([r for r in rows if r.get("research_task") != "unibo_six_choice_facies"], guard)
    if len(rows) != 480 or len(actual_facies) != 352 or len(validation) != 50:
        raise ValueError("Unexpected frozen sample counts")
    if Path(config["runtime"]).resolve() != runtime.resolve():
        raise ValueError("Runtime differs from parent")
    return rows, validation, guard


def prepare(parent, runtime, output):
    parent_config = json.loads((parent / "run_config.json").read_text(encoding="utf-8"))
    parent_result = json.loads((parent / "result.json").read_text(encoding="utf-8"))
    if parent_result["adapter_sha256"] != PARENT_SHA:
        raise ValueError("Parent completion identity differs")
    rows, validation, guard = validate_parent_rows(parent_config, parent, runtime)
    diagnostic = json.loads((parent / "post_hoc_training_fit.json").read_text(encoding="utf-8"))
    train_by_id = {r["metadata"]["record_id"]: r for r in rows}
    train_fit = [train_by_id[r["record_id"]] for r in diagnostic["answers"]]
    if (len(train_fit) != 24 or len({r["metadata"]["record_id"] for r in train_fit}) != 24
            or Counter(r["conversations"][1]["value"] for r in train_fit) != Counter(dict.fromkeys(source.CODES, 4))):
        raise ValueError("Training-fit probe identities differ")
    for name, records in (("train", rows), ("validation_facies", validation), ("validation_lithology", guard), ("train_fit_probe", train_fit)):
        source.write_rows(output / f"{name}.jsonl", records)
    dependencies = {str(p): source.digest(p) for p in (
        Path(source.__file__), ROOT / "scripts/train_qwen_source_pilot.py",
        ROOT / "scripts/train_lithology_facies_continuation.py", ROOT / "app/infrastructure/preference_loss.py",
        ROOT / "app/infrastructure/open_dataset_import.py", runtime / "train_qwen3vl.py")}
    hashes = {str(parent / name): source.digest(parent / name) for name in (
        "run_config.json", "result.json", "train.jsonl", "validation_facies.jsonl", "validation_lithology.jsonl",
        "post_hoc_training_fit.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors")}
    config = {"recipe": RECIPE, "criteria": CRITERIA, "parent": str(parent), "runtime": str(runtime),
              "base": parent_config["base"], "base_revision": parent_config["base_revision"],
              "base_weights_identity": parent_config["base_weights_identity"],
              "base_config_tokenizer_sha256": parent_config["base_config_tokenizer_sha256"],
              "package_versions": source.package_versions(), "python_version": sys.version,
              "script_sha256": source.digest(__file__), "dependencies": dependencies, "parent_files": hashes,
              "manifest_sha256": {p.name: source.digest(p) for p in output.glob("*.jsonl")},
              "facies_class_weights": source.facies_weights(list(r for r in rows if r.get("research_task") == "unibo_six_choice_facies")),
              "parent_adapter_sha256": PARENT_SHA, "role": "isolated_noncommercial_facies_discrimination_research",
              "source_license": "CC BY-NC-SA 4.0", "source_url": parent_config["source_url"],
              "attribution": parent_config["attribution"], "prompt": source.PROMPT,
              "unique_training_images": len(rows), "facies_images": 352, "replay_images": 128,
              "validation_images": 50, "lithology_guard_images": 64,
              "prepared_test_pixels_opened": 0, "validation_is_development": True,
              "notes": ["Same data, no new independent photos", "No conclusions about causal advantage of loss alone: objective and training duration both change",
                        "Six source facies only, no mapping to92 project facies", "Train-fit24 is diagnostic, not validation",
                        "All pairwise probes reuse the50 validation images", "No production activation or parent acceptance changes"]}
    source.write_json(output / "run_config.json", config)
    shutil.copy2(__file__, output / "training_script_snapshot.py")
    shutil.copy2(parent / "LICENSE_SOURCES.md", output / "LICENSE_SOURCES.md")
    source.write_json(output / "acceptance_review.json", {
        "schema": "qwen_acceptance_review_v1", "status": "rejected_for_general_assistant_task_scope",
        "installed_in_application": False, "geologist_approved": False,
        "research_evaluation_status": "prepared", "source_license": config["source_license"],
        "reason_codes": ["classification_task_not_general_core_description", "noncommercial_source_restrictions"],
        "permitted_role": config["role"]})
    return config, rows, validation, guard, train_fit


def restore(output, parent, runtime):
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    if (config["recipe"] != RECIPE or config["criteria"] != CRITERIA or config["script_sha256"] != source.digest(__file__)
            or source.digest(output / "training_script_snapshot.py") != config["script_sha256"]):
        raise ValueError("Resume recipe or implementation changed")
    if config["package_versions"] != source.package_versions() or config["python_version"] != sys.version:
        raise ValueError("Resume environment changed")
    if Path(config["parent"]).resolve() != parent or Path(config["runtime"]).resolve() != runtime:
        raise ValueError("Resume paths differ")
    for paths in (config["dependencies"], config["parent_files"], config["base_config_tokenizer_sha256"]):
        for path, expected in paths.items():
            if source.digest(path) != expected:
                raise ValueError("Resume file changed: " + path)
    for name, expected in config["manifest_sha256"].items():
        if source.digest(output / name) != expected:
            raise ValueError("Resume manifest changed: " + name)
    if (Path(config["base"]) / "model.safetensors").stat().st_size != config["base_weights_identity"]["bytes"]:
        raise ValueError("Resume base weight size changed")
    validate_parent_rows(json.loads((parent / "run_config.json").read_text(encoding="utf-8")), parent, runtime)
    return (config, *(source.read_rows(output / f"{name}.jsonl") for name in (
        "train", "validation_facies", "validation_lithology", "train_fit_probe")))


def train_fit_evaluation(model, processor, records, code_ids, output, stage):
    from PIL import Image
    answers = []
    for row in records:
        with Image.open(row["image"]) as opened:
            picture = opened.convert("RGB")
            picture.thumbnail((300, 300))
        answers.append({"record_id": row["metadata"]["record_id"], "source_group": row["metadata"]["source_group"],
                        "expected_code": row["conversations"][1]["value"], **source.infer_facies(model, processor, picture, code_ids)})
    result = {"role": "same24_training_fit_diagnostic_not_validation", "answers": answers,
              "metrics": source.classification_metrics([r["expected_code"] for r in answers], [r["forced_code"] for r in answers])}
    source.write_json(output / f"{stage}_train_fit.json", result)
    return result["metrics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT.parent / "vlm_core_v1")
    parser.add_argument("--parent", type=Path, default=ROOT.parent / "vlm_core_v1/outputs/core_vlm_visual_facies_20261030_lora")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.runtime, args.parent, args.output = (p.resolve() for p in (args.runtime, args.parent, args.output))
    if args.resume:
        if (args.output / "result.json").exists():
            raise ValueError("Completed experiment cannot resume")
        config, records, validation, guard, train_fit = restore(args.output, args.parent, args.runtime)
        checkpoint = complete_checkpoint(args.output / "trainer")
    else:
        if args.output.exists():
            raise ValueError("Choose a new output directory")
        args.output.mkdir(parents=True)
        config, records, validation, guard, train_fit = prepare(args.parent, args.runtime, args.output)
        checkpoint = None
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "unique_images": len(records), "epochs": RECIPE["epochs"]}), flush=True)
        return
    from scripts.train_qwen_source_pilot import configure_offline_environment, _deny_network
    configure_offline_environment()
    sys.addaudithook(_deny_network)
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, set_seed
    sys.path.insert(0, str(args.runtime))
    from train_qwen3vl import CorePhotoDataset, SingleImageCollator
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 6.5 * 1024**3:
        raise RuntimeError("Requires CUDA with at least6.5GiB free memory")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    set_seed(RECIPE["seed"])
    processor = AutoProcessor.from_pretrained(config["base"], local_files_only=True)
    class WeightedDataset(CorePhotoDataset):
        def __init__(self, rows, processor, max_image_side, weights):
            super().__init__(rows, processor, max_image_side)
            self.weights = weights

        def __getitem__(self, index):
            item = super().__getitem__(index)
            row = self.records[index]
            facies = row.get("research_task") == "unibo_six_choice_facies"
            code = row["conversations"][1]["value"]
            item["task_index"] = torch.tensor(source.CODES.index(code) if facies else -1)
            item["example_weight"] = torch.tensor(self.weights.get(code, 1.0) if facies else 1.0)
            return item
    dataset = WeightedDataset(records, processor, 300, config["facies_class_weights"])
    code_map = source.preflight(processor, dataset, records, validation, args.output)
    code_ids = tuple(code_map[code] for code in source.CODES)
    model = Qwen3VLForConditionalGeneration.from_pretrained(config["base"], local_files_only=True,
        dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    model = PeftModel.from_pretrained(model, args.parent / "adapter", is_trainable=True)
    def log(event, **values):
        data = {"event": event, **values}
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(json.dumps(data, ensure_ascii=True), flush=True)
    before = source.evaluate(model, processor, "before", validation, guard, code_map, records, args.output, log)
    before_fit = train_fit_evaluation(model, processor, train_fit, code_map, args.output, "before")
    source.write_json(args.output / "before_identity.json", {"fresh_evaluation": True, "adapter_sha256": PARENT_SHA, "checkpoint_loaded": False})
    model.train()
    model.config.use_cache = False
    class DiscriminationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            task_index, weight = inputs.pop("task_index"), inputs.pop("example_weight")
            if task_index.numel() != 1 or weight.numel() != 1:
                raise ValueError("Batch size must stay one")
            loss, outputs, _ = discrimination_loss(model, inputs, code_ids, int(task_index.item()))
            loss = loss * weight.reshape(()).to(loss.device)
            return (loss, outputs) if return_outputs else loss
    class Callback(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            log("training_loss", step=state.global_step, **(logs or {}))
    arguments = TrainingArguments(output_dir=str(args.output / "trainer"), num_train_epochs=RECIPE["epochs"],
        per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=RECIPE["learning_rate"],
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", logging_steps=10, save_strategy="steps", save_steps=50, save_total_limit=2,
        remove_unused_columns=False, dataloader_num_workers=0, report_to="none", seed=RECIPE["seed"], data_seed=RECIPE["seed"],
        warmup_steps=math.ceil(math.ceil(len(records) / 4) * RECIPE["epochs"] * .03), disable_tqdm=True)
    trainer = DiscriminationTrainer(model=model, args=arguments, train_dataset=dataset,
        data_collator=SingleImageCollator(processor.tokenizer.pad_token_id), processing_class=processor, callbacks=[Callback()])
    trainer.model_accepts_loss_kwargs = False
    started = perf_counter()
    log("training_started", epochs=RECIPE["epochs"], unique_images=len(records), checkpoint=str(checkpoint) if checkpoint else None)
    training = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    adapter = args.output / "adapter"
    trainer.save_model(str(adapter))
    processor.save_pretrained(adapter)
    model.gradient_checkpointing_disable()
    model.zero_grad(set_to_none=True)
    if trainer.optimizer:
        trainer.optimizer.state.clear()
        trainer.optimizer = None
    trainer.lr_scheduler = None
    gc.collect()
    torch.cuda.empty_cache()
    after = source.evaluate(model, processor, "candidate", validation, guard, code_map, records, args.output, log)
    after_fit = train_fit_evaluation(model, processor, train_fit, code_map, args.output, "candidate")
    for stage in ("before", "candidate"):
        source.write_json(args.output / f"{stage}_pairwise_probes.json", pairwise_probes(json.loads((args.output / f"{stage}_facies_answers.json").read_text(encoding="utf-8"))))
    new = after["facies"]["forced_choice"]
    gates = research_gates(before, after)
    result = {"before": before, "candidate": after, "before_train_fit": before_fit, "candidate_train_fit": after_fit,
              "gates": gates, "training_metrics": training.metrics, "seconds_training_and_after_eval": perf_counter() - started,
              "parent_adapter_sha256": PARENT_SHA, "adapter_sha256": source.digest(adapter / "adapter_model.safetensors"),
              "status": "research_evaluated_not_activated", "installed_in_application": False, "geologist_approved": False,
              "source_license": config["source_license"], "role": config["role"]}
    source.write_json(args.output / "result.json", result)
    review = json.loads((args.output / "acceptance_review.json").read_text(encoding="utf-8"))
    review.update({"research_evaluation_status": "complete", "research_gates": gates, "adapter_sha256": result["adapter_sha256"]})
    source.write_json(args.output / "acceptance_review.json", review)
    source.write_json(adapter / "acceptance_review.json", review)
    shutil.copy2(args.output / "LICENSE_SOURCES.md", adapter / "LICENSE_SOURCES.md")
    log("complete", facies_correct=new["correct"], facies_macro_f1=new["macro_f1"],
        lithology_correct=after["raw_lithology_guard"]["correct"], all_research_gates=all(gates.values()))


if __name__ == "__main__":
    main()
