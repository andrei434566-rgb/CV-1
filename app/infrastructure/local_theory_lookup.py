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


@lru_cache(maxsize=4)
def _load_cards(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lookup_theory_answer(question: str, cards_path: Path, *, include_sources: bool = False) -> str | None:
    """Answer only when wording is close enough to identify a specific card."""
    if not cards_path.is_file():
        return None
    query = _normalize(question)
    if len(query) < 12:
        return None
    corpus = _load_cards(str(cards_path))
    best_card = None
    best_score = 0.0
    query_terms = {term for term in query.split() if len(term) >= 5}
    for card in corpus.get("cards", []):
        for wording in [card["question"], *card.get("aliases", [])]:
            candidate = _normalize(wording)
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
        cards = corpus.get("cards", [])
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
