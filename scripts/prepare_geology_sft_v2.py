"""Freeze expanded source-checked theory, mixed context modes and a new challenge."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.infrastructure.geology_reference import format_grounded_question
from app.infrastructure.local_vlm_runner import prepare_geology_reference_chat
from scripts.prepare_qwen_theory_round2 import check_partitions, normalized_question


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_row(card, question, sources, *, training, mode):
    return {
        "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": card["answer"]}],
        "metadata": {
            "record_id": card["id"] + "__" + mode,
            "record_type": "text_only_knowledge", "source_group": "geology_sft_v2",
            "sources": sources, "training_approved": training, "geology_approved": False,
            "split": "train" if training else "validation", "target_language": "ru",
            "topic": card["topic"], "original_question": card["question"],
            "evaluation_mode": mode, "rubric": {
                "required_facts": card.get("required_facts", []),
                "forbidden_facts": card.get("forbidden_facts", []),
            },
            "review_status": "source_checked_assistant_authored_not_geologist_reviewed",
            "scope": "Theory only; not photo/facies labels or an expert quality certificate",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "vlm_source/geology_sft_v2_20261027")
    parser.add_argument("--previous", type=Path, default=ROOT / "open_datasets/qwen_grounded_v1_20261027")
    parser.add_argument("--output", type=Path, default=ROOT / "open_datasets/qwen_geology_sft_v2_20261027")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite a frozen experiment")
    cards_path = args.source / "training_cards.json"
    registry_path = args.source / "source_registry.json"
    challenge_path = args.source / "evaluation_challenge.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8-sig"))["cards"]
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))["sources"]
    challenge_document = json.loads(challenge_path.read_text(encoding="utf-8-sig"))
    if (challenge_document.get("record_role") != "evaluation_only_never_train"
            or challenge_document.get("freeze_status") != "frozen_before_training_launch"
            or challenge_document.get("geologist_reviewed") is not False
            or any(challenge_document.get("provenance", {}).get(key) is not False for key in (
                "training_question_files_read", "previous_evaluation_question_files_read", "model_outputs_read"))):
        raise ValueError("Challenge lacks its independently authored, frozen evaluation provenance")
    notes_provenance = challenge_document["provenance"]
    if digest(ROOT / notes_provenance["source_notes_path"]) != notes_provenance["source_notes_sha256"]:
        raise ValueError("Challenge was authored against different reference notes")
    challenge = [{**item, "answer": item["reference_answer"], "topic": item["concept"],
                  "required_facts": item["rubric"]["required_facts"],
                  "forbidden_facts": item["rubric"]["forbidden_facts"]}
                 for item in challenge_document["items"]]
    if len(challenge) != 16 or not cards:
        raise ValueError("Expected a fixed sixteen-question challenge and nonempty train cards")
    old_training = read_rows(args.previous / "mixed.jsonl")
    old_theory = read_rows(args.previous / "holdout_theory.jsonl")
    dialogue = read_rows(args.previous / "holdout_conversation.jsonl")
    images = read_rows(args.previous / "val_visual.jsonl")
    if len(old_training) != 124 or len(dialogue) != 4 or len(images) != 8:
        raise ValueError("Unexpected prior training or regression pool")
    # Keep historical holdouts in known project qwen* datasets protected too.
    protected = {normalized_question(row["metadata"].get("original_question", row["conversations"][0]["value"]))
                 for row in [*old_theory, *dialogue]}
    historical_holdouts = sorted(args.previous.parent.glob("qwen*/holdout*.jsonl"))
    for path in historical_holdouts:
        for row in read_rows(path):
            if row.get("image") or not row.get("conversations"):
                continue
            question = row.get("metadata", {}).get("original_question", row["conversations"][0]["value"])
            protected.add(normalized_question(question))
    challenge_questions = [normalized_question(card["question"]) for card in challenge]
    if len(set(challenge_questions)) != len(challenge_questions):
        raise ValueError("Duplicate new challenge questions")
    protected.update(challenge_questions)
    omitted_previous = [row["metadata"]["record_id"] for row in old_training
                        if normalized_question(row["metadata"].get("original_question", row["conversations"][0]["value"])) in protected]
    training = [copy.deepcopy(row) for row in old_training if row["metadata"]["record_id"] not in omitted_previous]
    prior_raw = {normalized_question(row["metadata"].get("original_question", row["conversations"][0]["value"]))
                 for row in training}
    if prior_raw & set(challenge_questions):
        raise ValueError("A newly authored challenge question already occurs in previous training")
    seen, omitted = set(), []
    for card in cards:
        if (card.get("split") != "train" or card.get("source_checked") is not True
                or card.get("geologist_reviewed") is not False):
            raise ValueError("Train cards require explicit source checking and separate expert status")
        question_key = normalized_question(card["question"])
        if question_key in protected or question_key in prior_raw:
            omitted.append({"id": card["id"], "reason": "protected_or_previous_exact_question"})
            continue
        if question_key in seen:
            raise ValueError("Duplicate new training question: " + card["id"])
        seen.add(question_key)
        if not card.get("support_text") or not card.get("answer") or not card.get("source_ids"):
            raise ValueError("Missing source support or answer: " + card["id"])
        sources = []
        for identity in card["source_ids"]:
            source = registry[identity]
            if not source.get("license") or not str(source.get("url", "")).startswith("https://"):
                raise ValueError("Missing source reuse conditions: " + identity)
            sources.append(dict(source, source_id=identity))
        training.append(make_row(card, card["question"], sources, training=True, mode="closed_book_train"))
        prompt = format_grounded_question(card["question"], [{"id": card["id"], "text": card["support_text"]}])
        training.append(make_row(card, prompt, sources, training=True, mode="supplied_context_train"))
    theory, retrieval_audit = [], []
    for card in challenge:
        # The original question is evaluated twice, not counted as two independent scenarios.
        source_ids = {identity for evidence in card["evidence"] for identity in evidence["source_ids"]}
        sources = [dict(challenge_document["sources"][identity], source_id=identity) for identity in sorted(source_ids)]
        theory.append(make_row(card, card["question"], sources, training=False, mode="new_closed_book"))
        prompt, hits, note_ids = prepare_geology_reference_chat({"operation": "chat", "question": card["question"]}, [])
        row = make_row(card, prompt, sources, training=False, mode="new_application_reference")
        row["metadata"]["reference_note_ids"] = note_ids
        row["metadata"]["reference_found"] = bool(note_ids)
        if note_ids:
            theory.append(row)
        retrieval_audit.append({"id": card["id"], "note_ids": note_ids,
                                "context_added": prompt != card["question"], "source_hits": hits})
    closed = [copy.deepcopy(row) for row in old_theory if row["metadata"].get("evaluation_mode") == "closed_book_regression"]
    if len(closed) != 4:
        raise ValueError("Missing four prior closed-book regression questions")
    theory.extend(closed)
    partitions = check_partitions(training, [*theory, *dialogue])
    raw_train = {normalized_question(row["metadata"].get("original_question", row["conversations"][0]["value"])) for row in training}
    raw_validation = {normalized_question(row["metadata"].get("original_question", row["conversations"][0]["value"])) for row in [*theory, *dialogue]}
    if raw_train & raw_validation:
        raise ValueError("Raw train/validation question overlap")
    args.output.mkdir(parents=True, exist_ok=True)
    exports = {"mixed.jsonl": training, "holdout_theory.jsonl": theory,
               "holdout_conversation.jsonl": dialogue, "val_visual.jsonl": images}
    for name, rows in exports.items():
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    (args.output / "system_prompt.txt").write_bytes((args.previous / "system_prompt.txt").read_bytes())
    protocol = {
        "schema": "qwen_geology_sft_v2_protocol", "status": "frozen_before_training", "date": "2026-10-27",
        "training_rows": len(training), "new_source_cards_used": len(seen), "previous_training_rows": len(old_training) - len(omitted_previous),
        "omitted_previous_training_due_to_protected_questions": omitted_previous,
        "training_modes": dict(Counter(row["metadata"].get("evaluation_mode", "previous_train") for row in training)),
        "omitted_new_cards": omitted, "new_scenarios": len(challenge), "new_evaluation_conditions": "closed book and reference where retrieval succeeds",
        "new_reference_rows": sum(bool(item["note_ids"]) for item in retrieval_audit),
        "old_closed_book_guards": 4, "conversation_guards": len(dialogue), "image_guards": len(images),
        "theory_evaluation_rows": len(theory), "retrieval_audit": retrieval_audit, "partition_checks": partitions,
        "raw_training_validation_question_overlap": 0,
        "recipe": {"fresh_base": True, "epochs": 3, "learning_rate": 1e-4, "lora_targets": "all-text-linear",
                   "rank": 8, "alpha": 16, "dropout": .05, "batch_size": 1, "gradient_accumulation": 4,
                   "max_sequence_tokens": 1024, "theory_max_new_tokens": 160, "image_max_new_tokens": 24,
                   "seed": 20261027, "precision": "FP16", "gradient_checkpointing": "nonreentrant"},
        "source_sha256": {path.name: digest(path) for path in [cards_path, registry_path, challenge_path]},
        "historical_holdout_sha256": {str(path.relative_to(args.previous.parent)): digest(path) for path in historical_holdouts},
        "manifest_sha256": {name: digest(args.output / name) for name in [*exports, "system_prompt.txt"]},
        "limitations": ["New scenarios were authored separately before training, but concepts and sources overlap.",
                        "Two conditions per question are paired observations, not independent samples.",
                        "This is not an independent-well geological exam or evidence of 95% facies accuracy.",
                        "Assistant source checking is not geologist approval.",
                        "A reference miss remains a miss; the expected passage is never inserted by an oracle.",
                        "Recipe selected after previous experiments; one fixed final candidate, no validation checkpoint selection.",
                        "Train source questions appear with and without supplied context intentionally, never across train/validation.",
                        "Active application weights stay unchanged until candidate evaluation."],
    }
    (args.output / "trial_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: protocol[key] for key in ["training_rows", "new_source_cards_used", "new_scenarios", "theory_evaluation_rows", "omitted_new_cards", "partition_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
