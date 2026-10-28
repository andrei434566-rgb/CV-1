"""Export checked TRAIN answers as exact offline reference, never model weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUP = "geology_sft_v2_20261027"
SOURCE = ROOT / "vlm_source" / GROUP
OUTPUT = ROOT / "app/resources/theory/theory_cards_sft_v2.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path = SOURCE) -> dict:
    cards_path, registry_path = source / "training_cards.json", source / "source_registry.json"
    document = json.loads(cards_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["sources"]
    if document.get("source_checked") is not True or document.get("geologist_reviewed") is not False:
        raise ValueError("Source checking must remain separate from expert approval")
    source_cards = document["cards"]
    if len(source_cards) != 110:
        raise ValueError("Expected the fixed 110 TRAIN cards")
    cards, sources, ids, questions = [], {}, set(), set()
    for original in source_cards:
        if (original.get("split") != "train" or original.get("source_checked") is not True
                or original.get("geologist_reviewed") is not False
                or not original.get("id") or original["id"] in ids
                or not isinstance(original.get("question"), str) or not original["question"].strip()
                or original["question"] in questions
                or not isinstance(original.get("answer"), str) or not original["answer"].strip()
                or not original.get("source_ids")):
            raise ValueError("Only unique, checked TRAIN questions and complete source answers may be exported")
        ids.add(original["id"])
        questions.add(original["question"])
        exported_sources = []
        for source_id in original["source_ids"]:
            record = registry[source_id]
            if (record.get("source_checked") is not True or record.get("geologist_reviewed") is not False
                    or not record.get("license") or not record.get("title")
                    or not str(record.get("url", "")).startswith("https://")):
                raise ValueError("Every reference requires checked source attribution and reuse conditions")
            exported_id = "geo_sft_v2_" + source_id
            sources[exported_id] = copy.deepcopy(record)
            exported_sources.append(exported_id)
        cards.append({
            "id": original["id"], "question": original["question"], "answer": original["answer"],
            "aliases": [], "sources": exported_sources, "holdout": False, "exact_only": True,
            "metadata": {"origin_manifest": "vlm_source/" + GROUP + "/training_cards.json",
                         "origin_split": "train", "origin_source_group": GROUP,
                         "review_status": "source_checked_original_russian_paraphrase_not_geologist_reviewed",
                         "geology_approved": False, "reference_scope": "general_theory"},
        })
    return {
        "schema": "kern_source_checked_reference_supplement_v1",
        "delivery": "offline_reference_answers_not_model_weight_training",
        "review_status": "source_checked_original_russian_paraphrases_not_geologist_reviewed",
        "geologist_reviewed": False,
        "scope": "Exact TRAIN question lookup only; does not certify VLM weights, photo descriptions or facies.",
        "source_manifest_sha256": digest(cards_path), "source_registry_sha256": digest(registry_path),
        "counts": {"cards": len(cards), "sources": len(sources), "validation_cards_included": 0},
        "sources": sources, "cards": cards,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"cards": len(result["cards"]), "sha256": digest(args.output), "delivery": result["delivery"]}))


if __name__ == "__main__":
    main()
