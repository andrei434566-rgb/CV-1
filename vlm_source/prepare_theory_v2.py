"""Build a sourced Russian theory curriculum without turning text into photo labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CARDS = ROOT / "data" / "knowledge" / "theory_cards_v2.json"
TRAIN = ROOT / "data" / "knowledge" / "theory_training_v2.jsonl"
HOLDOUT = ROOT / "data" / "knowledge" / "theory_holdout_v2.jsonl"
HOLDOUT_IDS = {
    "v2_contact_geometry",
    "v2_homogeneous_ichnofabric",
    "v2_induration",
    "v2_facies_taxonomy",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=CARDS)
    args = parser.parse_args()
    corpus = json.loads(args.cards.read_text(encoding="utf-8"))
    sources = {entry["id"]: entry for entry in corpus["sources"]}
    assert len(sources) == len(corpus["sources"])
    assert len({card["id"] for card in corpus["cards"]}) == len(corpus["cards"])

    train, holdout = [], []
    for card in corpus["cards"]:
        source = sources[card["source"]]
        row = {
            "conversations": [
                {"from": "human", "value": card["question"]},
                {"from": "gpt", "value": card["answer"]},
            ],
            "metadata": {
                "record_id": card["id"],
                "record_type": "text_only_knowledge",
                "review_status": "source_checked_assistant_paraphrase_not_geologist_reviewed",
                "training_approved": card["id"] not in HOLDOUT_IDS,
                "geology_approved": False,
                "target_language": "ru",
                "source_group": "core_theory_v2",
                "sources": [{
                    "source_id": source["id"],
                    "url": source["url"],
                    "locator": card["locator"],
                    "license": source["license"],
                }],
                "scope": "Terminology and description procedure only; never an image-to-facies ground truth label.",
            },
        }
        (holdout if card["id"] in HOLDOUT_IDS else train).append(row)

    for path, records in ((TRAIN, train), (HOLDOUT, holdout)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
    print(json.dumps({"train": len(train), "holdout": len(holdout), "sources": len(sources)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
