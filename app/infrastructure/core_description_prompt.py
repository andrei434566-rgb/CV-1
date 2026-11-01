"""Compact observation-first prompts shared by individual and batched photos."""
import re

DESCRIPTION_SYSTEM = (
    "Ты геолог-седиментолог. Описывай керн на русском языке. Разделяй наблюдения и гипотезы. "
    "Каждое изображение — отдельный образец, описание которого нельзя копировать с соседнего."
)


def description_prompt(context: str = "", whole_column: bool = False) -> str:
    subject = "целого столбика керна" if whole_column else "одного интервала керна"
    return (
        f"Опиши {subject}. Верни ТОЛЬКО короткий JSON, без markdown, ровно 6 строковых ключей: "
        "description, lithology, texture, facies_name, facies_association, depositional_environment. "
        "description: 1–2 предложения, до 25 слов, только видимые цвет, слоистость, контакты и трещины. "
        "lithology: предполагаемая порода, до 4 слов. "
        "texture: видимая текстура, до 6 слов. "
        "facies_name: предполагаемая фация, до 6 слов. "
        "facies_association: группа фаций, до 5 слов. "
        "depositional_environment: среда накопления, до 5 слов. "
        "Если фация или среда не определяются по фото, напиши 'Не установлена по фото'. "
        "Не описывай фон, ящик и этикетки. Не выдумывай признаки. "
        "Контекст: " + str(context)[:300]
    )


def check_interpretation_roles(fields: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Reject clear field-role errors; this is not geological validation."""
    values, notes = dict(fields), []
    def normalized(value):
        return re.sub(r"[^а-яёa-z ]", "", value.casefold()).strip()
    rock = normalized(values.get("lithology", "")).removeprefix("предположительно ").strip()
    rock_only = {"песчаник", "алевролит", "аргиллит", "суглинок", "глина", "песок", "известняк", "доломит"}
    for key in ("facies_name", "facies_association"):
        value = normalized(values.get(key, "")).removeprefix("предположительно ").strip()
        if value.startswith("не установлен"):
            values[key] = "Не установлена по фото"
            continue
        if value and (value == rock or value in rock_only):
            title = "Фация" if key == "facies_name" else "Ассоциация фаций"
            notes.append(f"{title}: модель подставила название породы вместо фациальной интерпретации")
            values[key] = "Не установлена по фото"
        elif (key == "facies_association" and re.fullmatch(
                r"(?:слойчат\w*|слоист\w*|массивн\w*) (?:аргиллит\w*|алевролит\w*|песчаник\w*)(?: фация)?", value)):
            notes.append("Ассоциация фаций: модель вернула только породу и её текстуру")
            values[key] = "Не установлена по фото"
    environment = normalized(values.get("depositional_environment", ""))
    if environment.startswith("не установлен"):
        values["depositional_environment"] = "Не установлена по фото"
    elif environment in {"разделение по горизонтали", "разделение пород"}:
        notes.append("Обстановка: модель вернула описание строения вместо среды накопления")
        values["depositional_environment"] = "Не установлена по фото"
    elif re.search("вулканич|метаморфич", environment) and not re.search("туф|вулкан|пирокласт|метаморф", rock):
        notes.append("Модель предложила происхождение без соответствующего определения породы; нужна проверка обстановки")
        values["depositional_environment"] = "Не установлена по фото"
    return values, notes
