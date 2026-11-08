"""Conservative review hints, never automatic geological corrections."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from app.infrastructure.description_template import PHOTO_UNRESOLVED, UNKNOWN_FACIES, observation


@dataclass(frozen=True)
class ReviewHint:
    code: str
    fields: tuple[str, ...]
    message: str

    def to_dict(self):
        return asdict(self)


def review_description(attributes: dict[str, str]) -> list[ReviewHint]:
    values = {key: observation(value)[0].casefold().replace("ё", "е") for key, value in attributes.items()}
    texture = values.get("Текстура", "")
    bedding = values.get("Слоистость", "")
    bio = values.get("Биотурбация", "")
    hints = []
    if texture == "массивная" and bedding and not any(t in bedding for t in ("не вид", "не различ", "не наблюд")):
        hints.append(ReviewHint("massive_and_bedded", ("Текстура", "Слоистость"),
            "Выбраны массивная текстура и слоистость. Уточните, относятся ли они к разным прослоям или масштабам наблюдения."))
    if texture == "биотурбационная" and any(t in bio for t in ("не обнаруж", "не наблюд", "отсутств", "нет видим")):
        hints.append(ReviewHint("bioturbation_presence", ("Текстура", "Биотурбация"),
            "Биотурбационная текстура указана вместе с отсутствием видимых следов. Проверьте выбранные признаки и участок фото."))
    if values.get("Название породы") in {"аргиллит", "глинистая порода"} and values.get("Зернистость") in {"крупнозернистый", "среднезернистый"}:
        hints.append(ReviewHint("grain_fraction_scope", ("Название породы", "Зернистость"),
            "Уточните, к какой фракции или прослою относится выбранная зернистость глинистой породы. Это запрос проверки, а не запрет сочетания."))
    return hints


DIAGNOSTIC_FIELDS = ("Название породы", "Зернистость", "Слоистость", "Текстура", "Биотурбация")


def facies_candidates(attributes, examples, *, assumptions=(), limit=3):
    """Rank only facies with verified examples and >=2 matching observations.

    The score is a count of agreements, not a calibrated probability. No
    catalogue defaults, inferred absence or colour-only facies prediction.
    """
    query = {key: observation(attributes.get(key))[0].casefold() for key in DIAGNOSTIC_FIELDS
             if key not in assumptions and not observation(attributes.get(key))[1]}
    ranked = {}
    for example in examples:
        if not example.get("description_reviewed") or not example.get("facies_reviewed"):
            continue
        label = example.get("facies", "")
        target = example.get("observations", {})
        matched, conflicts, missing = [], [], []
        for field in DIAGNOSTIC_FIELDS:
            q, ref = query.get(field), observation(target.get(field))[0].casefold()
            if not q or not ref or PHOTO_UNRESOLVED.casefold() in (q, ref):
                missing.append(field)
            elif q == ref:
                matched.append(field)
            else:
                conflicts.append(field)
        if len(matched) < 2 or len(conflicts) >= len(matched) or str(label).casefold() in UNKNOWN_FACIES:
            continue
        key = (label, str(example.get("facies_index", "")))
        row = {"facies": label, "name": example.get("facies_name") or label, "index": key[1],
               "matched": matched, "conflicts": conflicts, "missing": missing,
               "example_id": example["id"], "score": len(matched) - len(conflicts),
               "score_is_probability": False}
        if key not in ranked or row["score"] > ranked[key]["score"]:
            ranked[key] = row
    return sorted(ranked.values(), key=lambda row: (-row["score"], row["facies"], row["index"]))[:limit]
