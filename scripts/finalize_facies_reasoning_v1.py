"""Generate an auditable CPU-only report after a completed blind A/B review.

This utility does not train, convert, activate, or publish a model. The decision
is explicitly supplied by the reviewer; no score automatically approves it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / "vlm_core_v1/outputs/core_vlm_facies_reasoning_v1_v2_retry_lora"
DATA = ROOT / "open_datasets/qwen_facies_reasoning_v1_v2"
PUB = ROOT / "vlm_source/facies_reasoning_v1/reports"
BASE = "89644892e4d85e24eaac8bacfd4f463576704203"
STAGES = ("baseline", "candidate")
CATEGORIES = ("correct", "partial", "incorrect", "uncertain")
GROUPS = {"new_facies_formal": 11, "new_facies_exploratory": 1,
          "closed_book_regression": 4, "conversation": 4}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_review(blind, key, review):
    expected = {row["record_id"]: row for row in blind}
    if len(blind) != 20 or len(expected) != 20 or set(expected) != set(key):
        raise ValueError("Expected 20 unique blind pairs with a complete key")
    reviewed = review.get("rows")
    if not isinstance(reviewed, list) or not review.get("reviewer"):
        raise ValueError("A named reviewer and complete rows are required")
    counts, flags, completed = defaultdict(Counter), defaultdict(Counter), Counter()
    seen, decoded = set(), []
    for row in reviewed:
        identity = row.get("record_id")
        if identity not in expected or identity in seen:
            raise ValueError("Missing, duplicate or unexpected review ID: " + str(identity))
        seen.add(identity)
        group = expected[identity]["evaluation_mode"]
        if group == "new_facies_reasoning":
            group = "new_facies_exploratory" if identity == "FRV1-09__new_facies_reasoning" else "new_facies_formal"
        if group not in GROUPS:
            raise ValueError("Unexpected evaluation group: " + str(group))
        if set(key[identity]) != {"A", "B"} or set(key[identity].values()) != set(STAGES):
            raise ValueError("Invalid blinded answer key")
        for letter in ("A", "B"):
            result = row.get(letter, {})
            if (result.get("category") not in CATEGORIES or type(result.get("completed")) is not bool
                    or not isinstance(result.get("reason"), str) or not result["reason"].strip()
                    or not isinstance(result.get("flags", []), list)
                    or any(not isinstance(f, str) for f in result.get("flags", []))):
                raise ValueError("Invalid rating for " + identity + "/" + letter)
            stage = key[identity][letter]
            counts[group, stage][result["category"]] += 1
            flags[group, stage].update(result.get("flags", []))
            completed[group, stage] += int(result["completed"])
            decoded.append({"record_id": identity, "condition": stage, "group": group, **result})
    if seen != set(expected):
        raise ValueError("Blind review does not cover every pair")
    summary, failure_flags = {}, {}
    for group, size in GROUPS.items():
        summary[group], failure_flags[group] = {}, {}
        for stage in STAGES:
            values = counts[group, stage]
            if sum(values.values()) != size:
                raise ValueError("Unexpected review group size: " + group)
            summary[group][stage] = {"rows": size, **{category: values[category] for category in CATEGORIES},
                                     "completed": completed[group, stage]}
            failure_flags[group][stage] = dict(flags[group, stage])
    return summary, failure_flags, decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-confirmed", action="store_true", required=True)
    parser.add_argument("--decision", choices=("pending", "rejected", "theory_candidate"), required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--reason-code", action="append", default=[])
    args = parser.parse_args()

    protocol, config, metrics = read(DATA / "trial_protocol.json"), read(RUN / "run_config.json"), read(RUN / "metrics.json")
    if (protocol.get("training_rows") != 484 or protocol.get("excluded_from_formal_scoring") != ["FRV1-09"]
            or config.get("status") != "candidate_requires_geological_validation" or config.get("train_rows") != 484
            or config.get("epochs") != 2 or config.get("learning_rate") != 1e-4
            or config.get("lora_targets_recipe") != "all-text-linear"):
        raise ValueError("Run is incomplete or does not match the frozen protocol")
    for name, expected_hash in protocol["manifest_sha256"].items():
        if digest(DATA / name) != expected_hash:
            raise ValueError("Manifest changed: " + name)
    pub_protocol = PUB / "trial_protocol.json"
    if pub_protocol.exists() and digest(pub_protocol) != digest(DATA / "trial_protocol.json"):
        raise ValueError("Published frozen protocol differs")
    raw = {stage: rows(RUN / (stage + "_responses.jsonl")) for stage in STAGES}
    indexed = {stage: {row["record_id"]: row for row in data} for stage, data in raw.items()}
    if (any(len(raw[stage]) != 28 or len(indexed[stage]) != 28 for stage in STAGES)
            or set(indexed["baseline"]) != set(indexed["candidate"])):
        raise ValueError("Expected 28 complete, unique and paired evaluation responses")
    for identity in indexed["baseline"]:
        if any(indexed["baseline"][identity][k] != indexed["candidate"][identity][k]
               for k in ("question", "reference", "kind")):
            raise ValueError("Before/after evaluation mismatch: " + identity)
    blind, key, review = read(RUN / "blind_review.json"), read(RUN / "review_key.json"), read(RUN / "review_blind_ratings.json")
    counts, flags, decoded = decode_review(blind, key, review)
    for row in blind:
        identity = row["record_id"]
        for letter in ("A", "B"):
            source = indexed[key[identity][letter]][identity]
            if row[letter] != source["response"] or row[letter + "_truncated"] != source["truncated"]:
                raise ValueError("Blind review is inconsistent with raw responses: " + identity)
    progress = rows(RUN / "progress.jsonl")
    steps = max((row.get("step", 0) for row in progress if row.get("event") == "training_log"), default=0)
    if steps != 242 or not any(row.get("event") == "finished" for row in progress):
        raise ValueError("Expected 242 completed optimizer steps")
    image_guard = {}
    for stage in STAGES:
        source_values = metrics[stage]["image_classification"]
        image_guard[stage] = {k: sum(row[k] for row in source_values.values())
                              for k in ("rows", "correct", "unknown_output", "truncated")}
        image_guard[stage]["by_source"] = source_values
        if image_guard[stage]["rows"] != 8:
            raise ValueError("Expected exactly eight image-class guards")
    status = {"pending": "candidate_requires_geological_validation", "rejected": "rejected_for_general_assistant",
              "theory_candidate": "candidate_for_theory_only_requires_validation"}[args.decision]
    weights = RUN / "adapter/adapter_model.safetensors"
    limitations = [
        "Assistant blind evaluation is not approval by a geologist.",
        "Eleven conceptual cases share concepts with training and do not represent independent wells.",
        "FRV1-09 is exploratory because the source audit found insufficient support for its combined features.",
        "Four old geology and four conversation guards, and the eight image guards, were used before.",
        "Text-only training does not establish visual facies or complete-description accuracy.",
        "Transformers FP16 results do not establish the quality of the application's native Q8 runtime.",
        "Raw model outputs include errors and must not be used as geological references or training targets.",
        "No app settings or selected model are changed by this utility."
    ]
    report = {"schema": "qwen_facies_reasoning_v1_result", "record_type": "model_evaluation_audit_not_geological_reference",
              "reviewed_utc": datetime.now(timezone.utc).isoformat(), "status": status, "decision": args.decision,
              "reason": args.reason, "reason_codes": args.reason_code, "installed_in_application": False,
              "geologist_approved": False, "geology_accuracy": None, "base_model": "Qwen/Qwen3-VL-2B-Instruct",
              "base_revision": BASE, "model_source_fingerprint": config["model_source_fingerprint"],
              "data": {k: protocol[k] for k in ("training_rows", "new_train_cards", "positive_extension_cards", "new_scenarios", "excluded_from_formal_scoring")},
              "training": {**metrics["training"], "optimizer_steps": steps}, "recipe": protocol["recipe"],
              "qualitative_counts": counts, "review_flags": flags, "decoded_review": decoded,
              "source_class_image_guard": image_guard,
              "theory_token_limit_truncations": {stage: metrics[stage]["theory"]["truncated"] for stage in STAGES},
              "evaluation_seconds": {stage: metrics[stage]["seconds"] for stage in STAGES},
              "adapter": {"relative_local_directory": "vlm_core_v1/outputs/" + RUN.name + "/adapter",
                          "sha256": digest(weights), "bytes": weights.stat().st_size, "binary_uploaded_to_git": False},
              "protocol_sha256": digest(DATA / "trial_protocol.json"), "limitations": limitations}
    config_portable = copy.deepcopy(config)
    config_portable["schema"] = "qwen_facies_reasoning_v1_portable_config"
    config_portable["model"] = "Qwen/Qwen3-VL-2B-Instruct@" + BASE
    for field in ("adapter", "metrics"):
        if field in config_portable:
            config_portable[field] = RUN.name + "/" + Path(config_portable[field]).name
    for value in config_portable["manifests"].values():
        value["path"] = DATA.name + "/" + Path(value["path"]).name
    if "baseline_reuse" in config_portable:
        config_portable["baseline_reuse"]["source_directory"] = "previous_failed_run_with_complete_baseline"
    config_portable["original_sha256"] = digest(RUN / "run_config.json")
    config_portable["publication_transformation"] = "Machine paths replaced; parameters and hashes retained."
    PUB.mkdir(parents=True, exist_ok=True)
    copied = ["baseline_responses.jsonl", "candidate_responses.jsonl", "candidate_metrics.json",
              "blind_review.json", "review_key.json", "review_blind_ratings.json"]
    for name in copied:
        shutil.copyfile(RUN / name, PUB / name)
    baseline_metrics_portable = read(RUN / "baseline_metrics.json")
    if "reuse_provenance" in baseline_metrics_portable:
        baseline_metrics_portable["reuse_provenance"]["source_directory"] = "previous_failed_run_with_complete_baseline"
    write(PUB / "baseline_metrics.json", baseline_metrics_portable)
    shutil.copyfile(DATA / "trial_protocol.json", pub_protocol)
    write(PUB / "run_config.json", config_portable)
    write(PUB / "result_summary.json", report)
    write(RUN / "result_summary.json", report)
    if args.decision != "pending":
        acceptance = {"schema": "qwen_acceptance_review_v1", "status": status, "reason": args.reason,
                      "reason_codes": args.reason_code, "geologist_approved": False, "installed_in_application": False,
                      "geology_accuracy": None, "review_scope": "11 formal and 1 exploratory facies scenarios; 4 geology, 4 dialogue, 8 image-class guards"}
        write(RUN / "acceptance_review.json", acceptance)
        write(PUB / "acceptance_review.json", acceptance)
    write(PUB / "evidence_provenance.json", {"schema": "qwen_facies_reasoning_v1_evidence_provenance",
          "record_type": "audit_evidence_not_training_truth", "copied_without_changes": {name: digest(RUN / name) for name in copied},
          "baseline_metrics_original_sha256": digest(RUN / "baseline_metrics.json"),
          "baseline_metrics_publication": "Removed local absolute source directory; content otherwise unchanged.",
          "protocol_sha256": digest(DATA / "trial_protocol.json"), "raw_model_outputs_contain_errors": True,
          "raw_images_included": False, "model_weights_included": False})
    labels = {"new_facies_formal": "11 новых фациальных сценариев", "new_facies_exploratory": "FRV1-09: только исследовательский",
              "closed_book_regression": "4 прежних геологических вопроса", "conversation": "4 реплики общения"}
    lines = ["# Локальное обучение фациальным связям", "", "Qwen3-VL-2B: 484 текстовые учебные строки, включая 70 новых карточек в двух режимах; две эпохи.", "",
             "| Проверка: верно / частично / неверно / неясно | База | Кандидат |", "|---|---:|---:|"]
    for group, label in labels.items():
        values = [" / ".join(str(counts[group][stage][category]) for category in CATEGORIES) for stage in STAGES]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", f"Классы исходных изображений: база {image_guard['baseline']['correct']}/8; кандидат {image_guard['candidate']['correct']}/8.",
              f"Решение: `{status}`. {args.reason}", "Активная модель приложения не изменена.", "",
              "Оценка проведена ассистентом вслепую, без проверки геологом. Новые сценарии — текстовые, их источники и концепции пересекаются с обучением.",
              "FRV1-09 исключён из формального подсчёта. Проверка не измеряет точность фаций по фото или качество на новых скважинах.",
              "Сырые ответы с возможными ошибками сохранены для аудита и не являются материалом для обучения.", ""]
    (PUB / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "counts": counts, "adapter_sha256": report["adapter"]["sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
