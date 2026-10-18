"""Build source-attributed Russian theory Q&A and short offline reference notes.

Text-only cards are educational material; they never become photo or facies labels.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KNOWLEDGE = ROOT / "data" / "knowledge"
APP_THEORY = ROOT.parent / "kern_analyzer_app" / "app" / "resources" / "theory"
CARDS = KNOWLEDGE / "theory_cards_v3.json"
OLD_TRAIN = KNOWLEDGE / "theory_training_v2.jsonl"
TRAIN = KNOWLEDGE / "theory_training_v3.jsonl"
HOLDOUT = KNOWLEDGE / "theory_holdout_v3.jsonl"
TOPICS = {
    "rocks": "Обломочные и другие осадочные породы",
    "carbonates": "Карбонатные породы и смешанные литотипы",
    "structures": "Слоистость, контакты и осадочные структуры",
    "bioturbation": "Биотурбация и ихнофабрика",
    "method": "Порядок описания и пределы определения по фото",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    corpus = json.loads(CARDS.read_text(encoding="utf-8"))
    sources = corpus["sources"]
    cards = corpus["cards"]
    ids = [card["id"] for card in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate card id")
    train, holdout = [], []
    for card in cards:
        if card["topic"] not in TOPICS or not card["sources"]:
            raise ValueError(f"Invalid topic or missing source: {card['id']}")
        if not card["question"].strip() or not card["answer"].strip() or "<image>" in card["question"]:
            raise ValueError(f"Invalid text-only card: {card['id']}")
        if any(source_id not in sources for source_id in card["sources"]):
            raise ValueError(f"Unknown source: {card['id']}")
        is_holdout = bool(card.get("holdout"))
        row = {
            "conversations": [
                {"from": "human", "value": card["question"]},
                {"from": "gpt", "value": card["answer"]},
            ],
            "metadata": {
                "record_id": card["id"],
                "record_type": "text_only_knowledge",
                "topic": card["topic"],
                "target_language": "ru",
                "training_approved": not is_holdout,
                "geology_approved": False,
                "review_status": "source_checked_original_russian_paraphrase_not_geologist_reviewed",
                "source_group": "core_theory_v3",
                "sources": [
                    {"source_id": source_id, "title": sources[source_id]["title"],
                     "url": sources[source_id]["url"]}
                    for source_id in card["sources"]
                ],
                "scope": "Theory Q&A only; never a ground-truth image description or facies label.",
            },
        }
        (holdout if is_holdout else train).append(row)

    # Keep the prior sourced curriculum while starting a fresh adapter. Its
    # previous LoRA was rejected, so the weights are deliberately not resumed.
    prior = [json.loads(line) for line in OLD_TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = prior + train
    write_jsonl(TRAIN, train)
    write_jsonl(HOLDOUT, holdout)

    APP_THEORY.mkdir(parents=True, exist_ok=True)
    (APP_THEORY / CARDS.name).write_text(CARDS.read_text(encoding="utf-8"), encoding="utf-8")
    for topic, title in TOPICS.items():
        topic_cards = [card for card in cards if card["topic"] == topic]
        used_sources = sorted({source_id for card in topic_cards for source_id in card["sources"]})
        lines = [f"# {title}", "", "Справочные ответы для локального поиска. Это теория, а не разметка фотографий.", ""]
        for card in topic_cards:
            lines.extend([f"## {card['question']}", "", card["answer"], "",
                          "Источники: " + ", ".join(card["sources"]), ""])
        lines.extend(["## Первоисточники", ""])
        for source_id in used_sources:
            source = sources[source_id]
            lines.append(f"- {source_id}: {source['title']} — {source['url']}")
        content = "\n".join(lines) + "\n"
        (KNOWLEDGE / f"theory_{topic}_ru.md").write_text(content, encoding="utf-8")
        (APP_THEORY / f"theory_{topic}_ru.md").write_text(content, encoding="utf-8")
    print(json.dumps({"training_rows": len(train), "new_cards": len(cards),
                      "holdout_rows": len(holdout), "sources": len(sources)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
