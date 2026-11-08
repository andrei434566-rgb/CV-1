"""Deterministic Russian core descriptions from explicit interval observations.

No facies defaults, image inference or language model is used here. A missing
observation never becomes an assertion that a feature is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Iterable, Mapping

SCHEMA = "cv1-core-description-template-v1"
ASSUMPTIONS_KEY = "__description_assumptions"
PHOTO_UNRESOLVED = "Не различимо по фото"
UNKNOWN_FACIES = {"", "не определена", "не распознана", "интервал yolo", "новый контур"}
REVIEW_FIELDS = ("Название породы", "Цвет", "Зернистость", "Слоистость", "Контакт")
_EMPTY = {"", "—", "-", "не выбрано", "не выбрана", "не определено", "не определена",
          "не установлена", "не установлен", "не установлено", "не определяется",
          "литология требует проверки", "литология не установлена", "требуется проверка"}


def observation(value: object) -> tuple[str, bool]:
    """Keep user wording, remove only explicit uncertainty and missing markers."""
    text = " ".join(str(value or "").split()).strip(" .;\n")
    assumed = bool(re.match(r"^предположительно\s*[:—-]?\s*", text, re.I))
    if assumed:
        text = re.sub(r"^предположительно\s*[:—-]?\s*", "", text, flags=re.I).strip()
    if text.casefold() in _EMPTY or re.match(r"^(не установлен[ао]?|требует проверки|требуется проверка)\b", text, re.I):
        return "", False
    return text, assumed


def read_assumptions(attributes: Mapping[str, str]) -> set[str]:
    try:
        fields = json.loads(attributes.get(ASSUMPTIONS_KEY, "[]"))
    except (ValueError, TypeError):
        fields = []
    return {v for v in fields if isinstance(v, str)} if isinstance(fields, list) else set()


@dataclass(frozen=True)
class DescriptionDraft:
    text: str
    missing: tuple[str, ...]
    assumed: tuple[str, ...]


def compose_description(
    attributes: Mapping[str, str],
    depth_from: float | None = None,
    depth_to: float | None = None,
    *,
    facies_name: str = "",
    assumptions: Iterable[str] = (),
) -> DescriptionDraft:
    """Assemble only supplied observations; catalogue criteria are not inputs."""
    assumed = set(assumptions) | read_assumptions(attributes)
    found: dict[str, str] = {}
    lines: list[str] = []
    unresolved = False
    try:
        if (depth_from is not None and depth_to is not None
                and math.isfinite(float(depth_from)) and math.isfinite(float(depth_to))
                and float(depth_to) > float(depth_from)):
            top = f"{float(depth_from):.3f}".rstrip("0").rstrip(".").replace(".", ",")
            base = f"{float(depth_to):.3f}".rstrip("0").rstrip(".").replace(".", ",")
            if top == base:
                top = f"{float(depth_from):.10f}".rstrip("0").rstrip(".").replace(".", ",")
                base = f"{float(depth_to):.10f}".rstrip("0").rstrip(".").replace(".", ",")
            lines.append(f"Интервал {top}–{base} м.")
    except (TypeError, ValueError):
        pass

    fields = (
        ("Название породы", "Порода"), ("Цвет", "Цвет"), ("Зернистость", "Зернистость"),
        ("Примесь другой фракции", "Примесь другой фракции"), ("Текстура", "Текстура"),
        ("Слоистость", "Слоистость"), ("Биотурбация", "Биотурбация"),
        ("Включения", "Включения"), ("Органические остатки", "Органические остатки"),
        ("Пустотное пространство. Трещиноватость", "Пустоты и трещиноватость"),
        ("Цемент", "Цемент"), ("Степень цементации", "Степень цементации"),
        ("Флюидонасыщение", "Флюидонасыщение"), ("Контакт", "Контакт"),
        ("Ориентация контакта", "Ориентация контакта"), ("Целостность", "Целостность керна"),
        ("Фациальная ассоциация", "Ассоциация фаций"),
        ("Обстановка седиментации", "Обстановка осадконакопления"),
        ("Гидродинамический режим", "Гидродинамический режим"),
    )
    for field, title in fields:
        value, inline_assumption = observation(attributes.get(field))
        if not value:
            continue
        if value == PHOTO_UNRESOLVED:
            unresolved = True
            continue
        if inline_assumption:
            assumed.add(field)
        found[field] = value
        # Standard grain-size choices refer to the noun 'зернистость'. Free
        # text and Latin ichnotaxa retain the specialist's exact spelling.
        if field == "Зернистость":
            value = {"Крупнозернистый": "Крупнозернистая", "Среднезернистый": "Среднезернистая",
                     "Мелкозернистый": "Мелкозернистая", "Тонкозернистый": "Тонкозернистая"}.get(value, value)
        prefix = "предположительно " if field in assumed else ""
        lines.append(f"{title}: {prefix}{value}.")
    facies, inline_assumption = observation(facies_name)
    if facies == PHOTO_UNRESOLVED:
        unresolved = True
    elif facies.casefold() not in UNKNOWN_FACIES:
        found["Название фации"] = facies
        if inline_assumption:
            assumed.add("Название фации")
        lines.append("Фация: " + ("предположительно " if "Название фации" in assumed else "") + facies + ".")
    if unresolved and not found:
        lines.append("На фото признаки керна не различимы.")
    missing = tuple(field for field in (*REVIEW_FIELDS, "Название фации") if field not in found)
    return DescriptionDraft(" ".join(lines) or "Признаки участка не заполнены.", missing,
                            tuple(sorted(assumed.intersection(found))))
