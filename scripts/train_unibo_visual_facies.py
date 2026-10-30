"""Offline, bounded research continuation for six genuine UniBo source facies.

The CC BY-NC-SA 4.0 source and derived adapter remain restricted research
artifacts. Nothing here installs weights, changes parent acceptance flags,
reads the sealed test split, or claims general geology/depth competence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.infrastructure.open_dataset_import import UNIBO_CLASSES
from scripts.train_lithology_facies_continuation import complete_checkpoint

CODES = tuple("ABCDEF")
LABELS = tuple(item["display_ru"] for item in UNIBO_CLASSES)
SOURCE_CODES = tuple(item["abbreviation"] for item in UNIBO_CLASSES)
LITHOLOGY = ("песчаник", "известняк", "глинистая порода", "не керн")
INITIAL_SHA256 = "be5392871e8bdcd3b3cd530e37bcd9df7a6a70226e0a5b92c8102db8006829cf"
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
RECIPE = {"epochs": 2, "learning_rate": 1e-5, "batch_size": 1,
          "gradient_accumulation_steps": 4, "fp16": True, "attention": "sdpa",
          "max_image_side": 300, "torch_threads": 4, "interop_threads": 1,
          "lora_rank": 8, "seed": 20261030, "replay_rows": 128,
          "save_steps": 50, "save_total_limit": 2, "max_sequence_tokens": 1024,
          "class_weighting": "facies=N/(6*n_class), raw lithology replay=1; batch1 loss multiplier"}
PROMPT = (
    "Определи одну исходную фацию на фрагменте фотографии голоценового керна "
    "из набора University of Bologna «Sedimentary Facies Analysis and Segmentation». "
    "Используй только следующие шесть исходных классов:\n"
    + "\n".join(f"{code}: {item['display_ru']} ({item['name']})" for code, item in zip(CODES, UNIBO_CLASSES))
    + "\nОтветь ровно одной латинской буквой A, B, C, D, E или F."
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def package_versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "Pillow"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def code_to_russian(code):
    return LABELS[CODES.index(code)] if code in CODES else None


def raw_code(answer):
    """Only surrounding whitespace is ignored; JSON/prose/multiple codes fail."""
    value = answer.strip()
    return value if value in CODES else None


def transform_facies(row):
    """Retain publisher provenance verbatim while replacing only the task."""
    result = deepcopy(row)
    answer = json.loads(row["conversations"][1]["value"])
    if set(answer) != {"facies_name"} or answer["facies_name"] not in LABELS:
        raise ValueError("Unknown original facies JSON target")
    index = LABELS.index(answer["facies_name"])
    if row["metadata"]["source_class_code"] != SOURCE_CODES[index]:
        raise ValueError("Original source code and Russian target differ")
    if row["metadata"]["source_class_display_ru"] != LABELS[index]:
        raise ValueError("Original source display and Russian target differ")
    result["original_conversations"] = deepcopy(row["conversations"])
    result["conversations"] = [{"from": "human", "value": "<image>\n" + PROMPT},
                               {"from": "gpt", "value": CODES[index]}]
    result["research_task"] = "unibo_six_choice_facies"
    return result


def validate_facies_split(train, validation, *, verify_images=True):
    """Reject S-proxy, original-photo, crop-pixel, file and identity leakage."""
    seen = {name: {} for name in ("record_id", "source_group", "source_sha256", "pixel_sha256", "image_sha256")}
    counts = {}
    for split, rows in (("train", train), ("val", validation)):
        if not rows:
            raise ValueError("Empty facies split")
        for row in rows:
            meta = row["metadata"]
            if meta.get("split") != split:
                raise ValueError("Unexpected split; sealed test is refused")
            if (meta.get("source_id") != "unibo7308" or meta.get("source_license") != "CC BY-NC-SA 4.0"
                    or meta.get("label_provenance") != "source_direct_mask_exact_palette"
                    or meta.get("usage_scope") != "noncommercial_research_only_no_automatic_deployment"
                    or meta.get("training_approved") is not (split == "train")
                    or meta.get("geology_approved") is not False or meta.get("depths_available") is not False):
                raise ValueError("Facies source approval, research license or scope differs")
            transform_facies(row)
            for name, values in seen.items():
                value = meta.get(name)
                if not value:
                    raise ValueError("Missing provenance field: " + name)
                if value in values:
                    if values[value] != split:
                        raise ValueError("Train/validation leakage: " + name)
                    if name in {"record_id", "pixel_sha256", "image_sha256"}:
                        raise ValueError("Duplicate crop or record: " + name)
                values[value] = split
            if verify_images:
                from PIL import Image
                if digest(row["image"]) != meta["image_sha256"]:
                    raise ValueError("Crop file hash differs")
                with Image.open(row["image"]) as opened:
                    image = opened.convert("RGB")
                    pixels = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
                    pixels.update(image.tobytes())
                if pixels.hexdigest() != meta["pixel_sha256"]:
                    raise ValueError("Crop pixel hash differs")
        counts[split] = dict(Counter(row["metadata"]["source_class_code"] for row in rows))
        if set(counts[split]) != set(SOURCE_CODES):
            raise ValueError("All six genuine source facies must be present in each split")
    return {"class_counts": counts,
            "source_groups": {split: sorted({row["metadata"]["source_group"] for row in rows})
                              for split, rows in (("train", train), ("val", validation))},
            "unique_train_pixel_hashes": len(train), "unique_val_pixel_hashes": len(validation),
            "train_val_disjoint": ["source_group", "source_sha256", "pixel_sha256", "image_sha256", "record_id"],
            "source_group_note": "S-prefix grouping proxy; not a verified borehole identity"}


def select_replay(rows, seed):
    rng, selected = random.Random(seed), []
    for label in LITHOLOGY:
        group = sorted((row for row in rows if row["conversations"][1]["value"] == label),
                       key=lambda row: row["metadata"]["record_id"])
        if len(group) < 32:
            raise ValueError("Insufficient raw lithology replay: " + label)
        selected.extend(deepcopy(rng.sample(group, 32)))
    rng.shuffle(selected)
    return selected


def validate_lithology(replay, guard):
    if len(replay) != 128 or Counter(row["conversations"][1]["value"] for row in replay) != Counter(dict.fromkeys(LITHOLOGY, 32)):
        raise ValueError("Replay must have 32 unchanged source targets per class")
    if len(guard) != 64:
        raise ValueError("The exact previous 64-row validation guard is required")
    for split, rows in (("train", replay), ("val", guard)):
        for row in rows:
            meta = row["metadata"]
            if (meta.get("label_provenance") != "source_direct_folder" or meta.get("well_split") != split
                    or meta.get("training_approved") is not (split == "train")
                    or row["conversations"][1]["value"] not in LITHOLOGY):
                raise ValueError("Only approved raw lithology labels and held-out guard rows are accepted")
            if digest(row["image"]) != meta["image_sha256"]:
                raise ValueError("Lithology image hash differs")
    for field in ("image_sha256", "well_id", "record_id"):
        if {r["metadata"][field] for r in replay} & {r["metadata"][field] for r in guard}:
            raise ValueError("Replay/guard leakage: " + field)
    if len({r["metadata"]["image_sha256"] for r in replay}) != 128:
        raise ValueError("Duplicate replay image")


def facies_weights(rows):
    counts = Counter(row["conversations"][1]["value"] for row in rows)
    if set(counts) != set(CODES):
        raise ValueError("Weights require all six source facies")
    return {code: len(rows) / (len(CODES) * counts[code]) for code in CODES}


def classification_metrics(expected, predicted, labels=CODES):
    if not expected or len(expected) != len(predicted) or any(value not in labels for value in expected):
        raise ValueError("Metrics require aligned nonempty known references")
    invalid = "INVALID"
    confusion = {label: dict.fromkeys((*labels, invalid), 0) for label in labels}
    for truth, guess in zip(expected, predicted):
        confusion[truth][guess if guess in labels else invalid] += 1
    recall, precision, f1 = {}, {}, {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        guesses = sum(confusion[other][label] for other in labels)
        recall[label] = tp / support if support else 0.0
        precision[label] = tp / guesses if guesses else 0.0
        f1[label] = 2 * tp / (support + guesses) if support + guesses else 0.0
    correct = sum(truth == guess for truth, guess in zip(expected, predicted))
    return {"total": len(expected), "correct": correct, "accuracy": correct / len(expected),
            "macro_f1": sum(f1.values()) / len(labels), "per_class_recall": recall,
            "per_class_precision": precision, "per_class_f1": f1,
            "expected_distribution": dict(Counter(expected)),
            "prediction_distribution": dict(Counter(value if value in labels else invalid for value in predicted)),
            "invalid_outputs": sum(value not in labels for value in predicted),
            "confusion_expected_rows_predicted_columns": confusion}


def facies_stage_metrics(answers, gray, training_targets):
    expected = [row["expected_code"] for row in answers]
    train_counts = Counter(training_targets)
    majority = max(CODES, key=lambda code: train_counts[code])
    oracle_majority = max(CODES, key=lambda code: expected.count(code))
    result = {"forced_choice": classification_metrics(expected, [row["forced_code"] for row in answers]),
              "raw_generation": classification_metrics(expected, [row["raw_code"] for row in answers]),
              "gray_forced_choice": classification_metrics(expected, [gray["forced_code"]] * len(expected)),
              "gray_raw_generation": classification_metrics(expected, [gray["raw_code"]] * len(expected)),
              "training_majority": {"code": majority, **classification_metrics(expected, [majority] * len(expected))},
              "validation_majority_reference": {"code": oracle_majority, **classification_metrics(expected, [oracle_majority] * len(expected))},
              "per_source_group": {}, "code_to_russian": dict(zip(CODES, LABELS)),
              "source_group_note": "S-prefix proxy groups, not verified boreholes"}
    for group in sorted({row["source_group"] for row in answers}):
        subset = [row for row in answers if row["source_group"] == group]
        result["per_source_group"][group] = {
            "forced_choice": classification_metrics([r["expected_code"] for r in subset], [r["forced_code"] for r in subset]),
            "raw_generation": classification_metrics([r["expected_code"] for r in subset], [r["raw_code"] for r in subset])}
    normal = result["forced_choice"]["accuracy"]
    result["forced_accuracy_minus_gray"] = normal - result["gray_forced_choice"]["accuracy"]
    result["forced_accuracy_minus_training_majority"] = normal - result["training_majority"]["accuracy"]
    result["forced_accuracy_minus_validation_majority"] = normal - result["validation_majority_reference"]["accuracy"]
    result["gray_note"] = "One identical 256x256 RGB(128,128,128) image and prompt evaluated once, then fixed prediction applied to all validation references"
    return result


def prepare(runtime, data, initial, output):
    raw_train, raw_val = read_rows(data / "train.jsonl"), read_rows(data / "val.jsonl")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if summary.get("research_only") is not True or summary.get("prepared_test_pixels_opened") != 0:
        raise ValueError("Research provenance or sealed-test guarantee missing")
    split_audit = validate_facies_split(raw_train, raw_val)
    if summary["counts"] != split_audit["class_counts"]:
        raise ValueError("Summary class counts differ from manifests")
    train, validation = [transform_facies(r) for r in raw_train], [transform_facies(r) for r in raw_val]
    replay_path = runtime / "data/mendeley_lithology_4000/train.jsonl"
    guard_path = runtime / "outputs/core_vlm_lithology_facies_20261030_v4_lora/validation_images.jsonl"
    replay, guard = select_replay(read_rows(replay_path), RECIPE["seed"]), read_rows(guard_path)
    validate_lithology(replay, guard)
    weights = facies_weights(train)
    records = train + replay
    random.Random(RECIPE["seed"]).shuffle(records)
    for name, rows in (("train", records), ("validation_facies", validation), ("validation_lithology", guard)):
        write_rows(output / f"{name}.jsonl", rows)
    shutil.copy2(data / "summary.json", output / "source_preparation_summary.json")
    base = runtime / f"model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/{BASE_REVISION}"
    base_files = sorted(path for path in base.iterdir() if path.suffix in {".json", ".txt", ".jinja"})
    base_preprocessing_hashes = {str(path): digest(path) for path in base_files}
    if not (base / "model.safetensors").is_file():
        raise FileNotFoundError("The fixed local base model snapshot is required")
    if digest(initial / "adapter_model.safetensors") != INITIAL_SHA256:
        raise ValueError("Initial parent adapter checksum differs from the audited v4 adapter")
    adapter_config = json.loads((initial / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config.get("r") != 8 or adapter_config.get("peft_type") != "LORA":
        raise ValueError("Expected the existing rank-eight LoRA")
    dependencies = {str(path): digest(path) for path in (
        ROOT / "scripts/train_qwen_source_pilot.py", ROOT / "scripts/train_lithology_facies_continuation.py",
        ROOT / "app/infrastructure/preference_loss.py", ROOT / "app/infrastructure/open_dataset_import.py",
        runtime / "train_qwen3vl.py")}
    config = {"recipe": RECIPE, "base": str(base), "base_revision": BASE_REVISION,
              "runtime": str(runtime), "data": str(data), "initial_adapter": str(initial),
              "package_versions": package_versions(), "python_version": sys.version,
              "base_config_tokenizer_sha256": base_preprocessing_hashes,
              "base_weights_identity": {"revision": BASE_REVISION, "file": "model.safetensors",
                                        "bytes": (base / "model.safetensors").stat().st_size,
                                        "note": "Fixed local revision; full base weight checksum not recomputed"},
              "initial_adapter_sha256": INITIAL_SHA256, "initial_adapter_config_sha256": digest(initial / "adapter_config.json"),
              "script_sha256": digest(__file__), "dependency_sha256": dependencies,
              "source_manifest_sha256": {str(p): digest(p) for p in (data / "train.jsonl", data / "val.jsonl", data / "summary.json", replay_path, guard_path)},
              "training_manifest_sha256": digest(output / "train.jsonl"),
              "facies_validation_manifest_sha256": digest(output / "validation_facies.jsonl"),
              "lithology_guard_manifest_sha256": digest(output / "validation_lithology.jsonl"),
              "source_license": "CC BY-NC-SA 4.0", "source_url": "https://amsacta.unibo.it/id/eprint/7308/",
              "attribution": "Andrea Di Martino and Alessandro Amorosi (2023), DOI 10.6092/unibo/amsacta/7308",
              "replay_source_license": "CC BY 4.0", "replay_source_url": "https://data.mendeley.com/datasets/b6rj29mrss/1",
              "prepared_test_pixels_opened": 0,
              "original_publisher_splits_are_not_sealed": summary.get("original_publisher_splits_are_not_sealed"),
              "original_publisher_split_note": summary.get("original_publisher_split_note"),
              "role": "restricted_noncommercial_source_facies_research_candidate",
              "status": "prepared_not_activated", "parent_acceptance_flags_changed": False,
              "code_to_russian": dict(zip(CODES, LABELS)), "source_codes": dict(zip(CODES, SOURCE_CODES)),
              "facies_class_weights": weights, "facies_train_tasks": len(train), "facies_validation_tasks": len(validation),
              "replay_tasks": len(replay), "lithology_guard_tasks": len(guard), "training_tasks": len(records),
              "unique_training_images": len({r["metadata"]["image_sha256"] for r in records}),
              "facies_train_distribution": dict(Counter(r["conversations"][1]["value"] for r in train)),
              "replay_distribution": dict(Counter(r["conversations"][1]["value"] for r in replay)),
              "split_audit": split_audit, "prompt": PROMPT,
              "evaluation": {"facies": "next-token six-way softmax and independent unconstrained max_new_tokens=8 greedy generation",
                             "raw_code_matching": "exact A-F after trimming surrounding whitespace only",
                             "lithology": "unchanged raw source question; native image thumbnail448, greedy max_new_tokens16, no repetition penalty",
                             "baseline": "fresh initial-adapter before and candidate after; no cached baseline",
                             "gray": "single 256x256 RGB(128,128,128) evaluation per stage"},
              "limitations": ["Six source-domain Holocene facies only; no general core description or depth labels",
                              "S-prefix split groups are proxies; neighbouring source crops remain correlated",
                              "Development validation, not sealed independent final test; no geology approval inferred",
                              "License restrictions follow source; no automatic activation or commercial deployment"]}
    if config["unique_training_images"] != len(records):
        raise ValueError("Training images must be unique; replay is not oversampled")
    write_json(output / "run_config.json", config)
    shutil.copy2(__file__, output / "training_script_snapshot.py")
    (output / "LICENSE_SOURCES.md").write_text(
        "# Restricted noncommercial research candidate\n\n" + config["attribution"]
        + ". Source: https://amsacta.unibo.it/id/eprint/7308/\n\n"
        "UniBo source and derivatives: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/. "
        "Crops, Russian translations and model adaptation are modifications. Preserve attribution and share-alike/noncommercial restrictions. "
        "This experiment does not change parent acceptance flags or activate weights.\n\n"
        "Mendeley raw lithology replay: Fatimah Alzubaidi (2021), DOI 10.17632/b6rj29mrss.1, CC BY 4.0.\n",
        encoding="utf-8")
    return config, records, validation, guard


def validate_resume(config, output, runtime, data, initial):
    if config["recipe"] != RECIPE or config["script_sha256"] != digest(__file__):
        raise ValueError("Resume cannot change the saved recipe or training implementation")
    if digest(output / "training_script_snapshot.py") != config["script_sha256"]:
        raise ValueError("Resume training snapshot checksum differs")
    if config["package_versions"] != package_versions() or config["python_version"] != sys.version:
        raise ValueError("Resume Python or ML package versions differ")
    for key, actual in (("runtime", runtime), ("data", data), ("initial_adapter", initial)):
        if Path(config[key]).resolve() != actual.resolve():
            raise ValueError("Resume path differs: " + key)
    for paths in (config["dependency_sha256"], config["source_manifest_sha256"], config["base_config_tokenizer_sha256"]):
        for path, sha in paths.items():
            if digest(path) != sha:
                raise ValueError("Resume source/dependency checksum differs: " + path)
    if (Path(config["base"]) / "model.safetensors").stat().st_size != config["base_weights_identity"]["bytes"]:
        raise ValueError("Resume local base model size differs")
    for name, key in (("train", "training_manifest_sha256"), ("validation_facies", "facies_validation_manifest_sha256"),
                      ("validation_lithology", "lithology_guard_manifest_sha256")):
        if digest(output / f"{name}.jsonl") != config[key]:
            raise ValueError("Resume prepared manifest checksum differs: " + name)
    if digest(initial / "adapter_model.safetensors") != config["initial_adapter_sha256"]:
        raise ValueError("Resume parent adapter checksum differs")
    if digest(initial / "adapter_config.json") != config["initial_adapter_config_sha256"]:
        raise ValueError("Resume parent adapter config checksum differs")


def prompt_inputs(processor, image, question):
    content = [{"type": "image", "image": image}, {"type": "text", "text": question.replace("<image>", "", 1).strip()}]
    rendered = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return processor(text=[rendered], images=[image], return_tensors="pt")


def preflight(processor, dataset, records, validation, output):
    """Independently reconstruct all prefixes and check the causal target j-1."""
    from PIL import Image
    from scripts.train_qwen_source_pilot import validate_training_tokens
    ids = {code: processor.tokenizer.encode(code, add_special_tokens=False) for code in CODES}
    if any(len(value) != 1 for value in ids.values()) or len({value[0] for value in ids.values()}) != 6:
        raise ValueError("Each A-F choice must be a distinct single tokenizer token")
    code_ids = {code: value[0] for code, value in ids.items()}
    audit = {"train_rows": len(records), "facies_validation_rows": len(validation), "code_token_ids": code_ids,
             "max_sequence_tokens": 0, "truncated_rows": 0, "supervised_prediction_position": "assistant token j uses hidden/logit position j-1",
             "prefix_checked_rows": 0, "facies_next_token_checked_rows": 0}
    # Validation targets are inspected only to verify token alignment, never used for loss.
    for index, row in enumerate(records + validation):
        if index < len(records):
            features = dataset[index]
        else:
            features = type(dataset)([row], processor, RECIPE["max_image_side"], {})[0]
        with Image.open(row["image"]) as opened:
            image = opened.convert("RGB")
            image.thumbnail((RECIPE["max_image_side"], RECIPE["max_image_side"]))
        prompt = prompt_inputs(processor, image, row["conversations"][0]["value"])
        full_ids, prefix_ids = features["input_ids"].tolist(), prompt["input_ids"][0].tolist()
        labels = features["labels"].tolist()
        counts = validate_training_tokens(full_ids, prefix_ids, labels,
            max_sequence_tokens=RECIPE["max_sequence_tokens"], identity=row["metadata"]["record_id"])
        if row.get("research_task") == "unibo_six_choice_facies":
            target = code_ids[row["conversations"][1]["value"]]
            if full_ids[len(prefix_ids)] != target or labels[len(prefix_ids)] != target:
                raise ValueError("Normal generation prefix does not directly predict the A-F training token")
            audit["facies_next_token_checked_rows"] += 1
        audit["max_sequence_tokens"] = max(audit["max_sequence_tokens"], counts["sequence_tokens"])
        audit["prefix_checked_rows"] += 1
    write_json(output / "token_preflight.json", audit)
    return code_ids


def infer_facies(model, processor, image, code_ids):
    import torch
    inputs = prompt_inputs(processor, image, PROMPT).to("cuda")
    with torch.inference_mode():
        output = model(**inputs, logits_to_keep=1, use_cache=False)
        if output.logits.ndim != 3 or output.logits.shape[:2] != (1, 1):
            raise ValueError("Qwen/PEFT next-token logits_to_keep=1 was not honored")
        logits = output.logits[0, -1, [code_ids[code] for code in CODES]].float()
        probabilities = logits.softmax(dim=-1).cpu().tolist()
        forced = CODES[int(logits.argmax().item())]
        del output, logits
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
    new_ids = generated[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    return {"forced_code": forced, "forced_russian": code_to_russian(forced),
            "probabilities_A_to_F": dict(zip(CODES, probabilities)), "raw_answer": answer,
            "raw_code": raw_code(answer), "raw_russian": code_to_russian(raw_code(answer)),
            "generation_tokens": new_ids.shape[1], "generation_reached_limit": new_ids.shape[1] >= 8}


def predict_lithology(model, processor, row):
    """Same algorithm as runtime evaluate_barnegat_binary_pilot.predict."""
    import torch
    from PIL import Image
    with Image.open(row["image"]) as opened:
        image = opened.convert("RGB")
        image.thumbnail((448, 448))
    inputs = prompt_inputs(processor, image, row["conversations"][0]["value"]).to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    return processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def evaluate(model, processor, stage, validation, guard, code_ids, records, output, log):
    from PIL import Image
    model.eval()
    model.config.use_cache = True
    answers, lithology = [], []
    gray = infer_facies(model, processor, Image.new("RGB", (256, 256), (128, 128, 128)), code_ids)
    write_json(output / f"{stage}_gray.json", gray)
    for index, row in enumerate(validation, 1):
        with Image.open(row["image"]) as opened:
            image = opened.convert("RGB")
            image.thumbnail((300, 300))
        answer = {"record_id": row["metadata"]["record_id"], "source_group": row["metadata"]["source_group"],
                  "image_sha256": row["metadata"]["image_sha256"], "expected_code": row["conversations"][1]["value"],
                  "expected_russian": code_to_russian(row["conversations"][1]["value"]),
                  **infer_facies(model, processor, image, code_ids)}
        answers.append(answer)
        write_json(output / f"{stage}_facies_answers.json", answers)
        log("evaluation", stage=stage, task="facies", row=index, total=len(validation),
            forced_correct=answer["forced_code"] == answer["expected_code"], raw_valid=answer["raw_code"] is not None)
    for index, row in enumerate(guard, 1):
        answer = predict_lithology(model, processor, row)
        # Keep the same exact normalized raw-target scoring as the parent run.
        predicted = answer.casefold().strip().strip(' .,:;!"«»')
        lithology.append({"record_id": row["metadata"]["record_id"], "well_id": row["metadata"]["well_id"],
                          "image_sha256": row["metadata"]["image_sha256"], "raw_answer": answer,
                          "expected": row["conversations"][1]["value"], "prediction": predicted})
        write_json(output / f"{stage}_lithology_answers.json", lithology)
        log("evaluation", stage=stage, task="raw_lithology_guard", row=index, total=len(guard),
            correct=predicted == lithology[-1]["expected"])
    targets = [r["conversations"][1]["value"] for r in records if r.get("research_task") == "unibo_six_choice_facies"]
    metrics = {"facies": facies_stage_metrics(answers, gray, targets),
               "raw_lithology_guard": classification_metrics([r["expected"] for r in lithology],
                                                              [r["prediction"] for r in lithology], LITHOLOGY)}
    write_json(output / f"{stage}_metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT.parent / "vlm_core_v1")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New research run directory; required")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.runtime, args.output = args.runtime.resolve(), args.output.resolve()
    data = (args.data or args.runtime / "data/unibo_vlm_facies_20261030").resolve()
    initial = (args.initial_adapter or args.runtime / "outputs/core_vlm_lithology_facies_20261030_v4_lora/adapter").resolve()
    checkpoint = None
    if args.resume:
        if (args.output / "result.json").exists():
            raise ValueError("Complete runs cannot resume; choose a new output directory")
        config = json.loads((args.output / "run_config.json").read_text(encoding="utf-8"))
        validate_resume(config, args.output, args.runtime, data, initial)
        records, validation, guard = (read_rows(args.output / f"{name}.jsonl")
                                      for name in ("train", "validation_facies", "validation_lithology"))
        # Check original untouched source manifests, then restored image/guard provenance.
        validate_facies_split(read_rows(data / "train.jsonl"), read_rows(data / "val.jsonl"))
        validate_lithology([r for r in records if r.get("research_task") != "unibo_six_choice_facies"], guard)
        checkpoint = complete_checkpoint(args.output / "trainer")
    else:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new run directory")
        args.output.mkdir(parents=True)
        config, records, validation, guard = prepare(args.runtime, data, initial, args.output)
    if args.prepare_only:
        print(json.dumps({"status": config["status"], "training_tasks": len(records),
                          "facies_val": len(validation), "raw_guard": len(guard)}, ensure_ascii=True), flush=True)
        return
    from scripts.train_qwen_source_pilot import configure_offline_environment, _deny_network, sft_supervised_token_loss
    configure_offline_environment()
    sys.addaudithook(_deny_network)
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, set_seed
    sys.path.insert(0, str(args.runtime))
    from train_qwen3vl import CorePhotoDataset, SingleImageCollator
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this trainer does not fall back to CPU")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    free, _total = torch.cuda.mem_get_info()
    if free < 6.5 * 1024**3:
        raise RuntimeError("At least 6.5 GiB free CUDA memory is required; close the active VLM first")
    set_seed(RECIPE["seed"])
    processor = AutoProcessor.from_pretrained(config["base"], local_files_only=True)
    class WeightedDataset(CorePhotoDataset):
        def __init__(self, rows, processor, max_image_side, weights):
            super().__init__(rows, processor, max_image_side)
            self.weights = weights

        def __getitem__(self, index):
            result = super().__getitem__(index)
            row = self.records[index]
            weight = self.weights.get(row["conversations"][1]["value"], 1.0) if row.get("research_task") == "unibo_six_choice_facies" else 1.0
            result["example_weight"] = torch.tensor(weight, dtype=torch.float32)
            return result
    dataset = WeightedDataset(records, processor, 300, config["facies_class_weights"])
    code_ids = preflight(processor, dataset, records, validation, args.output)
    model = Qwen3VLForConditionalGeneration.from_pretrained(config["base"], local_files_only=True,
        dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    model = PeftModel.from_pretrained(model, initial, is_trainable=True)
    def log(event, **values):
        data = {"event": event, **values}
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(json.dumps(data, ensure_ascii=True), flush=True)
    log("run_started", resume=args.resume, tasks=len(records), unique_images=config["unique_training_images"])
    # Always evaluate the initial adapter anew, including resume; checkpoint is loaded by Trainer afterwards.
    before = evaluate(model, processor, "before", validation, guard, code_ids, records, args.output, log)
    write_json(args.output / "before_parent_adapter_identity.json", {"adapter_sha256": digest(initial / "adapter_model.safetensors"),
        "fresh_evaluation": True, "checkpoint_loaded": False})
    model.train()
    model.config.use_cache = False
    class TokenLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            weight = inputs.pop("example_weight")
            if weight.numel() != 1 or inputs["input_ids"].shape[0] != 1:
                raise ValueError("Class weighting is defined for fixed batch size one")
            loss, outputs = sft_supervised_token_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            loss = loss * weight.reshape(()).to(loss.device)
            return (loss, outputs) if return_outputs else loss
    class Callback(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            log("training_loss", step=state.global_step, **(logs or {}))
    arguments = TrainingArguments(output_dir=str(args.output / "trainer"), num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-5,
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", logging_steps=10, save_strategy="steps", save_steps=50, save_total_limit=2,
        remove_unused_columns=False, dataloader_num_workers=0, report_to="none", seed=RECIPE["seed"],
        data_seed=RECIPE["seed"], warmup_steps=math.ceil(math.ceil(len(records) / 4) * 2 * .03), disable_tqdm=True)
    trainer = TokenLossTrainer(model=model, args=arguments, train_dataset=dataset,
        data_collator=SingleImageCollator(processor.tokenizer.pad_token_id), processing_class=processor, callbacks=[Callback()])
    started = perf_counter()
    log("training_started", epochs=2, checkpoint=str(checkpoint) if checkpoint else None)
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
    after = evaluate(model, processor, "candidate", validation, guard, code_ids, records, args.output, log)
    result = {"before": before, "candidate": after, "training_loss_metrics": training.metrics,
              "seconds_training_and_after_evaluation": perf_counter() - started,
              "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
              "initial_adapter_sha256": INITIAL_SHA256, "role": config["role"],
              "status": "restricted_research_candidate_requires_review_not_activated",
              "parent_acceptance_flags_changed": False, "source_license": config["source_license"]}
    write_json(args.output / "result.json", result)
    log("complete", status=result["status"], before_facies_accuracy=before["facies"]["forced_choice"]["accuracy"],
        candidate_facies_accuracy=after["facies"]["forced_choice"]["accuracy"],
        candidate_macro_f1=after["facies"]["forced_choice"]["macro_f1"],
        raw_lithology_guard_correct=after["raw_lithology_guard"]["correct"])


if __name__ == "__main__":
    main()
