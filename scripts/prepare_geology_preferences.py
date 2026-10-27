"""Create train-only preference pairs from checked notes, never model validation answers."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_qwen_theory_round2 import check_partitions, normalized_question


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "vlm_source/grounded_theory_v1/preference_questions.json")
    parser.add_argument("--previous", type=Path, default=ROOT / "open_datasets/qwen_grounded_v1_20261027")
    parser.add_argument("--output", type=Path, default=ROOT / "open_datasets/qwen_geology_dpo_v1_20261027")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite a preference experiment")
    preferences = json.loads(args.source.read_text(encoding="utf-8"))
    previous = rows(args.previous / "mixed.jsonl")
    by_id = {row["metadata"]["record_id"]: row for row in previous}
    training = []
    for pair in preferences["cards"]:
        identity = pair["id"]
        row = copy.deepcopy(by_id[identity])
        chosen = pair.get("chosen", pair["answer"])
        if (chosen != row["conversations"][1]["value"] or not isinstance(pair.get("rejected"), str)
                or not pair["rejected"].strip() or pair["rejected"].strip() == chosen.strip()
                or pair.get("split") != "train"):
            raise ValueError("Invalid preference target: " + identity)
        role = pair.get("rejected_metadata", {}).get("training_role") or pair.get("training_role")
        if role != "rejected_false_statement_not_ground_truth":
            raise ValueError("Counterexample is not marked as rejected: " + identity)
        row["preference_rejected"] = pair["rejected"].strip()
        row["metadata"].update(preference_source="source_checked_authored_counterexample",
                               preference_rejection_reason=pair["rejection_reason"],
                               rejected_training_role=role)
        training.append(row)
    if len(training) != 64:
        raise ValueError("Expected the 64 fixed source-conditioned train questions")
    # Conversational guards are chosen from previous TRAIN rows only.
    conversation = [row for row in previous if row["metadata"]["record_type"] == "authored_conversation_not_geological_label"]
    if len(conversation) != 60:
        raise ValueError("Unexpected conversation training pool")
    bad_conversation = ["Сначала пришлите фото керна, иначе я не могу с вами общаться.",
                        "Я уже сохранил ваш проект и отправил отчёт, хотя вы этого не просили.",
                        "В предоставленной геологической справке нет данных, поэтому на приветствие не отвечу.",
                        "Я всегда безошибочен и не нуждаюсь в проверке геолога."]
    thanks = [row for row in conversation if re.search(r"спасиб|благодар", row["conversations"][0]["value"], re.I)][:4]
    others = [row for row in conversation if row not in thanks][:4]
    if len(thanks) != 4 or len(others) != 4:
        raise ValueError("Expected four thanks and four ordinary-conversation train guards")
    for index, original in enumerate([*thanks, *others]):
        row = copy.deepcopy(original)
        row["preference_rejected"] = bad_conversation[index % len(bad_conversation)]
        row["metadata"].update(preference_source="authored_ordinary_conversation_guard",
                               preference_rejection_reason="Unrequested action claim, irrelevant photo dependency, refusal of ordinary conversation, or infallibility",
                               rejected_training_role="rejected_false_statement_not_ground_truth")
        training.append(row)
    theory = rows(args.previous / "holdout_theory.jsonl")
    dialogue = rows(args.previous / "holdout_conversation.jsonl")
    validation = [*theory, *dialogue]
    partitions = check_partitions(training, validation)
    raw_train = {normalized_question(row["metadata"].get("original_question", row["conversations"][0]["value"])) for row in training}
    raw_val = {normalized_question(row["metadata"].get("original_question", row["conversations"][0]["value"])) for row in validation}
    if raw_train & raw_val:
        raise ValueError("Raw train and validation questions overlap")
    exports = {"mixed.jsonl": training, "holdout_theory.jsonl": theory,
               "holdout_conversation.jsonl": dialogue, "val_visual.jsonl": rows(args.previous / "val_visual.jsonl")}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in exports.items():
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
    (args.output / "system_prompt.txt").write_bytes((args.previous / "system_prompt.txt").read_bytes())
    protocol = {"schema": "geology_preference_pilot_v1", "status": "prepared_before_training",
                "training_pairs": len(training), "source_conditioned_pairs": 64, "ordinary_conversation_pairs": 8,
                "recipe": {"loss": "DPO", "beta": .1, "epochs": 2, "learning_rate": 5e-6,
                           "lora_rank": 8, "targets": "language-only q/v", "fresh_base": True, "seed": 20261027},
                "input_sha256": {str(args.source.relative_to(ROOT)): sha(args.source),
                                  "previous_train_manifest": sha(args.previous / "mixed.jsonl")},
                "manifest_sha256": {name: sha(args.output / name) for name in [*exports, "system_prompt.txt"]},
                "partition_checks": partitions,
                "limitations": ["Rejected statements are deliberate false counterexamples, never factual references",
                                "Targets are assistant checked against primary notes, not geologist reviewed",
                                "Evaluation is the same development/regression set as the prior grounded experiment",
                                "Recipe was selected after reviewing prior SFT results; no independent test or 95% accuracy claim",
                                "Raw validation model answers were not used to generate preference pairs",
                                "Only explicit preference mode may consume preference_rejected"]}
    (args.output / "trial_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"training_pairs": len(training), "validation_rows": len(validation), "partition_checks": partitions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
