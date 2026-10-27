"""Freeze source-conditioned Russian QA and fixed before/after evaluation inputs."""
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
from app.infrastructure.geology_reference import format_grounded_question, load_notes, retrieve_notes
from scripts.prepare_qwen_source_pilot import application_system_prompt
from scripts.prepare_qwen_theory_round2 import check_partitions, normalized_question


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_row(card, notes, sources, *, training, selected):
    evidence = [dict(sources[source], source_id=source) for note in selected for source in note["sources"]]
    return {"conversations": [{"from": "human", "value": format_grounded_question(card["question"], selected)},
                              {"from": "gpt", "value": card["answer"]}],
            "metadata": {"record_id": card["id"], "record_type": "text_only_knowledge",
                         "source_group": "geology_source_conditioned_v1", "sources": evidence,
                         "training_approved": training, "geology_approved": False, "target_language": "ru",
                         "review_status": "source_checked_original_paraphrase_not_geologist_reviewed",
                         "split": "train" if training else "validation",
                         "topic": card.get("topic") or notes[card.get("note_id", card.get("expected_note_id"))]["topic"],
                         "original_question": card["question"], "reference_note_ids": [note["id"] for note in selected],
                         "evaluation_mode": "source_conditioned" if not training else "training",
                         "example_kind": card.get("metadata", {}).get("kind", "source_supported"),
                         "required_facts": card.get("required_facts", []), "forbidden_facts": card.get("forbidden_facts", []),
                         "scope": "Context-conditioned theory, not memorized knowledge or an image/facies label"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "vlm_source/grounded_theory_v1")
    parser.add_argument("--previous", type=Path, default=ROOT / "open_datasets/qwen_theory_round4_20261027")
    parser.add_argument("--output", type=Path, default=ROOT / "open_datasets/qwen_grounded_v1_20261027")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite a frozen experiment")
    notes_path = args.source / "source_notes.json"
    corpus = load_notes(str(notes_path))
    notes = {note["id"]: note for note in corpus["notes"]}
    cards = json.loads((args.source / "training_questions.json").read_text(encoding="utf-8"))
    if cards["source_notes_sha256"] != sha(notes_path):
        raise ValueError("Training questions reference different source notes")
    validation = json.loads((args.source / "validation_questions.json").read_text(encoding="utf-8"))["cards"]
    old_train = read_rows(args.previous / "mixed.jsonl")
    old_holdout = read_rows(args.previous / "holdout_theory.jsonl")
    old_conversation_holdout = read_rows(args.previous / "holdout_conversation.jsonl")
    # Preserve every earlier held-out question as held out, even when it is not
    # part of this run. Metadata source URLs never become assistant instructions.
    sealed_questions = {normalized_question(row["conversations"][0]["value"])
                        for row in [*old_holdout, *old_conversation_holdout]}
    sealed_questions |= {normalized_question(card["question"]) for card in validation}
    training, omitted, seen = [], [], set()
    for card in cards["cards"]:
        if card.get("split") != "train":
            raise ValueError("Unexpected non-train question in training card file")
        normalized = normalized_question(card["question"])
        if normalized in sealed_questions:
            omitted.append(card["id"])
            continue
        if normalized in seen:
            raise ValueError("Duplicate raw training question")
        seen.add(normalized)
        note = notes[card.get("supplied_note_id", card["note_id"])]
        training.append(make_row(card, notes, corpus["sources"], training=True, selected=[note]))
    conversation = [copy.deepcopy(row) for row in old_train
                    if row["metadata"]["record_type"] == "authored_conversation_not_geological_label"]
    if len(conversation) != 60 or any(row.get("image") for row in conversation):
        raise ValueError("Expected sixty existing ordinary-conversation train rows")
    training.extend(conversation)
    new_validation, retrieval_audit = [], []
    for card in validation:
        selected = retrieve_notes(card["question"], notes_path, max_notes=2)
        if not selected:
            raise ValueError("No automatic reference match for validation scenario: " + card["id"])
        retrieval_audit.append({"id": card["id"], "expected_note_id": card["expected_note_id"],
                                "retrieved_ids": [note["id"] for note in selected],
                                "expected_found": card["expected_note_id"] in {note["id"] for note in selected}})
        new_validation.append(make_row(card, notes, corpus["sources"], training=False, selected=selected))
    closed_ids = {"v3_conglomerate_breccia", "v3_mud_shale", "v3_carbonate_mudstone", "v4_coal_vs_carbonaceous"}
    closed_guards = [copy.deepcopy(row) for row in old_holdout if row["metadata"]["record_id"] in closed_ids]
    if len(closed_guards) != 4:
        raise ValueError("Missing fixed closed-book guards")
    for row in closed_guards:
        row["metadata"]["evaluation_mode"] = "closed_book_regression"
    theory = [*new_validation, *closed_guards]
    dialogue = copy.deepcopy(old_conversation_holdout)
    images = read_rows(args.previous / "val_visual.jsonl")
    if len(images) != 8 or len(dialogue) != 4:
        raise ValueError("Unexpected previous regression subset")
    partitions = check_partitions(training, [*theory, *dialogue])
    exports = {"mixed.jsonl": training, "holdout_theory.jsonl": theory,
               "holdout_conversation.jsonl": dialogue, "val_visual.jsonl": images}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in exports.items():
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output / "system_prompt.txt").write_text(application_system_prompt(ROOT / "app/infrastructure/local_vlm_runner.py"), encoding="utf-8")
    protocol = {"schema": "grounded_geology_v1", "date": "2026-10-27", "status": "frozen_before_training",
                "train_rows": len(training), "grounded_train": len(training) - len(conversation), "conversation_train": len(conversation),
                "grounded_validation": len(new_validation), "closed_book_guards": 4, "dialogue_guards": len(dialogue), "image_guards": 8,
                "omitted_training_questions_due_to_holdout": omitted, "partition_checks": partitions,
                "retrieval_audit": retrieval_audit,
                "training_kinds": dict(Counter(row["metadata"].get("example_kind", "ordinary_conversation") for row in training)),
                "recipe": {"fresh_base": True, "epochs": 2, "learning_rate": 2e-5, "lora_targets": "qv", "lora_rank": 8,
                           "seed": 20261027, "max_sequence_tokens": 1024, "theory_max_new_tokens": 160, "image_max_new_tokens": 24},
                "source_files": {name: sha(args.source / name) for name in ["source_notes.json", "training_questions.json", "validation_questions.json"]},
                "manifest_sha256": {name: sha(args.output / name) for name in [*exports, "system_prompt.txt"]},
                "limitations": ["Source-conditioned QA measures reading supplied evidence, not closed-book factual knowledge",
                                "Training and new questions share reference notes and concepts; no independent expert accuracy claim",
                                "Regression questions/images were already used in development", "Assistant-authored targets, not geologist approval",
                                "Recipe also differs from round4 in targets/data; do not attribute a change causally to context alone",
                                "Raw model answers never become new training targets; active model remains unchanged until reviewed"]}
    (args.output / "trial_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: protocol[k] for k in ["train_rows", "grounded_train", "conversation_train", "grounded_validation", "closed_book_guards", "omitted_training_questions_due_to_holdout", "retrieval_audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
