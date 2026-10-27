"""Fast, source-attributed answers for close matches to curated theory cards."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[а-яёa-z0-9]+", value.casefold()))


def _question_without_source_suffix(value: str) -> str:
    """Remove only a trailing explicit citation request, not topic words."""
    cleaned = re.sub(
        r"[\s.!?;:,—-]+(?:пожалуйста[\s,]+)?"
        r"(?:дай(?:те)?|укажи(?:те)?|покажи(?:те)?|добавь(?:те)?|приведи(?:те)?)\s+"
        r"(?:источники|ссылки)(?:\s+и\s+(?:источники|ссылки))?"
        r"(?:[\s,]+пожалуйста)?[\s.!?]*$",
        "", value, flags=re.IGNORECASE,
    ).strip()
    return cleaned if cleaned else value


_GENERIC = {
    "какой", "каким", "какие", "почему", "можно", "нужно", "чтобы", "этого",
    "пород", "керна", "керне", "фото", "снимк", "отлич", "опред", "назва",
    "призн", "должн", "между", "этот", "таким", "тольк", "всегда", "будет",
    "распо", "разли", "почем", "обяза", "измер",
}
_SYNONYMS = {"шейл": "shale", "мадст": "mudst", "вакст": "wacks",
             "пакст": "packs", "грейн": "grain"}


def _stems(value: str) -> set[str]:
    result = set()
    for token in _normalize(value).split():
        if len(token) < 5:
            continue
        stem = token[:5]
        stem = _SYNONYMS.get(stem, stem)
        if stem not in _GENERIC:
            result.add(stem)
    return result


def _merge_reference_supplement(corpus: dict, supplement: dict, *,
                                expected_source_group: str = "theory_sources_20261025") -> dict:
    """Add checked train-only reference answers without changing fuzzy ranking."""
    if not isinstance(supplement, dict) or supplement.get("schema") != "kern_source_checked_reference_supplement_v1":
        return corpus
    source_registry = supplement.get("sources")
    new_cards = supplement.get("cards")
    if not isinstance(source_registry, dict) or not isinstance(new_cards, list):
        return corpus
    sources = dict(corpus.get("sources", {}))
    cards = list(corpus.get("cards", []))
    known_ids = {card.get("id") for card in cards}
    for card in new_cards:
        if not isinstance(card, dict):
            continue
        meta = card.get("metadata", {})
        if (not isinstance(meta, dict) or card.get("id") in known_ids or not card.get("id")
                or card.get("holdout") is not False or card.get("exact_only") is not True
                or card.get("aliases") or meta.get("origin_split") != "train"
                or meta.get("origin_source_group") != expected_source_group
                or meta.get("reference_scope") != "general_theory"
                or meta.get("geology_approved") is not False
                or not str(meta.get("review_status", "")).startswith(("source_checked", "user_requested_source_checked"))
                or not isinstance(card.get("question"), str) or not card["question"].strip()
                or not isinstance(card.get("answer"), str) or not card["answer"].strip()):
            continue
        source_ids = card.get("sources")
        if (not isinstance(source_ids, list) or not source_ids
                or not all(isinstance(value, str) and value.strip() for value in source_ids)):
            continue
        checked_sources = {}
        for source_id in source_ids:
            source = source_registry.get(source_id)
            if (not isinstance(source, dict) or not isinstance(source.get("title"), str)
                    or not source["title"].strip() or not isinstance(source.get("url"), str)
                    or not source["url"].startswith(("https://", "http://"))):
                break
            if source_id in sources and any(sources[source_id].get(key) != source.get(key) for key in ("title", "url")):
                break
            checked_sources[source_id] = source
        else:
            sources.update(checked_sources)
            cards.append(card)
            known_ids.add(card["id"])
    return {**corpus, "sources": sources, "cards": cards}


@lru_cache(maxsize=4)
def _load_cards(path: str) -> dict:
    cards_path = Path(path)
    corpus = json.loads(cards_path.read_text(encoding="utf-8"))
    if cards_path.name == "theory_cards_v5.json":
        for name, source_group in (("theory_cards_round2.json", "theory_sources_20261025"),
                                   ("theory_cards_round4.json", "theory_sources_20261027")):
            supplement = cards_path.with_name(name)
            if supplement.is_file():
                try:
                    corpus = _merge_reference_supplement(
                        corpus, json.loads(supplement.read_text(encoding="utf-8")),
                        expected_source_group=source_group,
                    )
                except (OSError, ValueError, TypeError):
                    # An unavailable supplement must not break established answers.
                    pass
    return corpus


def lookup_theory_answer(question: str, cards_path: Path, *, include_sources: bool = False) -> str | None:
    """Answer only when wording is close enough to identify a specific card."""
    if not cards_path.is_file():
        return None
    if re.search(
        r"\b(?:на|по|в)\s+(?:этом|данном|загруженн\w*|прикрепл[её]нн\w*|мо[её]м)\s+(?:фото|сним\w*|изображен\w*)\b",
        question.casefold(),
    ):
        return None
    query = _normalize(_question_without_source_suffix(question) if include_sources else question)
    if len(query) < 12:
        return None
    corpus = _load_cards(str(cards_path))
    best_card = None
    best_score = 0.0
    query_terms = {term for term in query.split() if len(term) >= 5}
    for card in corpus.get("cards", []):
        for wording in [card["question"], *card.get("aliases", [])]:
            candidate = _normalize(wording)
            if card.get("exact_only") and query != candidate:
                continue
            score = SequenceMatcher(None, query, candidate).ratio()
            candidate_terms = {term for term in candidate.split() if len(term) >= 5}
            overlap = len(query_terms & candidate_terms) / max(1, min(len(query_terms), len(candidate_terms)))
            if query == candidate:
                score = 1.0
            elif score < 0.84 or overlap < 0.65:
                continue
            if score > best_score:
                best_card, best_score = card, score
    if best_card is None:
        cards = [card for card in corpus.get("cards", []) if not card.get("exact_only")]
        query_stems = _stems(question)
        if len(query_stems) >= 2:
            frequency = Counter()
            card_terms = []
            for card in cards:
                terms = _stems(" ".join([card["question"], *card.get("aliases", [])]))
                card_terms.append((card, terms))
                frequency.update(terms)
            weights = {term: math.log((len(cards) + 1) / (count + 1)) + 1
                       for term, count in frequency.items()}
            query_weight = sum(weights.get(term, 3.0) for term in query_stems)
            ranked = []
            for card, terms in card_terms:
                shared = query_stems & terms
                if len(shared) < 2:
                    continue
                overlap = sum(weights.get(term, 3.0) for term in shared)
                coverage = overlap / query_weight
                precision = overlap / max(1.0, sum(weights.get(term, 3.0) for term in terms))
                ranked.append((0.75 * coverage + 0.25 * precision, coverage, precision, card))
            ranked.sort(key=lambda item: item[0], reverse=True)
            if ranked and ranked[0][0] >= 0.78 and (
                len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.14
                or (ranked[0][1] >= 0.95 and ranked[0][2] >= 0.45)
            ):
                best_card = ranked[0][3]
    if best_card is None:
        return None
    if not include_sources:
        return best_card["answer"]
    source_lines = []
    for source_id in best_card["sources"]:
        source = corpus["sources"][source_id]
        source_lines.append(f"{source['title']}: {source['url']}")
    return best_card["answer"] + "\n\nИсточники:\n" + "\n".join(source_lines)
