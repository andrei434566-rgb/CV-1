"""Build a source-attributed Russian theory experiment; never install a model."""
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
from scripts.prepare_qwen_source_pilot import application_system_prompt
from scripts.prepare_qwen_theory_round2 import check_partitions, normalized_question, record_id, validated_rows


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card_row(card, sources):
    training = card["split"] == "train"
    if card["split"] not in {"train", "validation"}:
        raise ValueError("Unknown card split")
    evidence = []
    for source_id in card["sources"]:
        source = sources[source_id]
        if not all(source.get(key) for key in ("title", "url", "license", "license_url", "sections")):
            raise ValueError("Incomplete attribution/license: " + source_id)
        evidence.append(dict(source, source_id=source_id, locator="; ".join(source["sections"])))
    return {"conversations": [{"from": "human", "value": card["question"]},
                              {"from": "gpt", "value": card["answer"]}],
            "metadata": {"record_id": card["id"], "record_type": "text_only_knowledge",
                         "source_group": "theory_round4_source_checked", "topic": card["topic"],
                         "target_language": "ru", "split": "train" if training else "validation",
                         "training_approved": training, "geology_approved": False,
                         "review_status": "source_checked_assistant_paraphrase_not_geologist_reviewed",
                         "sources": evidence, "rubric": card.get("rubric", {}),
                         "usage_scope": "Russian theory only; no image labels, invented depths or facies ground truth"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=ROOT / "vlm_source/theory_round4/new_source_cards.json")
    parser.add_argument("--previous", type=Path, default=ROOT / "open_datasets/qwen_theory_round2_20261025")
    parser.add_argument("--visual-validation", type=Path, default=ROOT / "open_datasets/qwen_visual_round2_20261025/val_visual.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "open_datasets/qwen_theory_round4_20261027")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite a prepared experiment")
    registry = json.loads(args.cards.read_text(encoding="utf-8-sig"))
    new = [card_row(card, registry["sources"]) for card in registry["cards"]]
    new_train = validated_rows([row for row in new if row["metadata"]["training_approved"]], training=True, theory=True)
    new_val = validated_rows([row for row in new if not row["metadata"]["training_approved"]], training=False, theory=True)
    old_train = read_jsonl(args.previous / "mixed.jsonl")
    old_val = read_jsonl(args.previous / "holdout_theory.jsonl")
    old_conversation_val = read_jsonl(args.previous / "holdout_conversation.jsonl")
    # Every earlier validation question stays excluded, including those omitted
    # from this run's small regression subset. No held-out answer becomes loss.
    excluded = {normalized_question(row["conversations"][0]["value"]) for row in [*old_val, *old_conversation_val, *new_val]}
    training, seen, duplicates = [], set(), []
    for row in [*new_train, *old_train]:
        row = copy.deepcopy(row)
        meta = row["metadata"]
        if row.get("image") or meta.get("split") != "train" or meta.get("training_approved") is not True:
            raise ValueError("Only approved text train rows may be reused")
        theory = meta.get("record_type") == "text_only_knowledge"
        validated_rows([row], training=True, theory=theory)
        question = normalized_question(row["conversations"][0]["value"])
        if question in excluded:
            raise ValueError("Training question occurs in validation: " + record_id(row))
        if question in seen:
            duplicates.append(record_id(row))
            continue
        seen.add(question)
        training.append(row)
    # Previous validation was used for recipe development; these are guards,
    # not a new independent benchmark or a numeric geological accuracy score.
    regression_ids = {"v3_conglomerate_breccia", "v3_mud_shale", "v3_carbonate_mudstone", "v4_coal_vs_carbonaceous"}
    regression = [row for row in old_val if record_id(row) in regression_ids]
    if len(regression) != len(regression_ids):
        raise ValueError("Missing fixed previous theory guard")
    theory_validation = [*new_val, *regression]
    conversation_validation = old_conversation_val[:4]
    visual = read_jsonl(args.visual_validation)
    by_class = {}
    for row in visual:
        meta = row["metadata"]
        key = (meta.get("dataset"), meta.get("target_label_ru") or meta.get("target_label"))
        by_class.setdefault(key, row)
    # Four Mendeley classes plus four DCID classes, with original source labels.
    selected = [row for (source, _), row in sorted(by_class.items(), key=lambda pair: str(pair[0])) if source == "mendeley"][:4]
    selected += [row for (source, _), row in sorted(by_class.items(), key=lambda pair: str(pair[0])) if source != "mendeley"][:4]
    if len(selected) != 8 or any(not Path(row["image"]).is_file() for row in selected):
        raise ValueError("Expected eight existing visual regression photographs")
    partitions = check_partitions(training, [*theory_validation, *conversation_validation])
    exports = {"mixed.jsonl": training, "holdout_theory.jsonl": theory_validation,
               "holdout_conversation.jsonl": conversation_validation, "val_visual.jsonl": selected}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in exports.items():
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output / "system_prompt.txt").write_text(application_system_prompt(ROOT / "app/infrastructure/local_vlm_runner.py"), encoding="utf-8")
    (args.output / "sources.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = {"schema": "qwen_theory_round4_20261027", "status": "prepared_not_trained",
                "data_counts": {"train": len(training), "new_train": len(new_train), "new_validation": len(new_val),
                                "theory_validation": len(theory_validation), "conversation_validation": len(conversation_validation),
                                "visual_regression": len(selected)},
                "training_types": dict(Counter(row["metadata"]["record_type"] for row in training)),
                "recipe": {"fresh_base": True, "epochs": 2, "learning_rate": 2e-5,
                           "lora_rank": 8, "lora_targets": "all-text-linear", "max_sequence_tokens": 1024,
                           "seed": 20261027, "max_new_tokens_theory": 200, "max_new_tokens_classification": 24},
                "partition_checks": partitions, "duplicate_questions_removed": duplicates,
                "evaluation": "Fixed before training; latest epoch only; no validation-based recipe/checkpoint selection in this run",
                "limitations": ["All theory targets were source checked by an assistant, not reviewed by a geologist",
                                "New scenarios share concepts/sources with training; they are validation, not a sealed independent test",
                                "Earlier theory, conversation and image guards have been used repeatedly during development",
                                "Text-only training supplies no new visual labels and cannot establish photo/facies accuracy",
                                "Adapter remains separate; promotion requires checking facts, dialogue and visual regression"],
                "input_hashes": {str(path.relative_to(ROOT)): digest(path) for path in [args.cards, args.previous / "mixed.jsonl",
                                   args.previous / "holdout_theory.jsonl", args.visual_validation]},
                "manifest_sha256": {name: digest(args.output / name) for name in [*exports, "system_prompt.txt", "sources.json"]}}
    (args.output / "trial_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **protocol["data_counts"], "partition_checks": partitions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
