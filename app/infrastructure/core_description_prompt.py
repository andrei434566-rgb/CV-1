"""Compact observation-first prompts shared by individual and batched photos."""
import re

DESCRIPTION_SYSTEM = (
    "Ты геолог-седиментолог. Описывай керн на русском языке. Разделяй наблюдения и гипотезы. "
    "Каждое изображение — отдельный образец, описание которого нельзя копировать с соседнего."
)


def description_prompt(context: str = "", whole_column: bool = False) -> str:
    subject = "целого столбика керна" if whole_column else "одного интервала керна"
    return (
        f"Показаны крупные планы {subject}. Это один образец; не описывай компоновку снимка. "
        "Верни только JSON с шестью строковыми полями:\n"
        "description: 2–3 законченных предложения, до 40 слов: цвет породы, форма и направление слойков, "
        "контакты, трещины и изменения по длине. Этикетки, номера образцов, ящик и фон не описывай.\n"
        "lithology: предположение о породе по видимой фактуре: песчаник, алевролит, аргиллит или переслаивание. "
        "По одному серому цвету размер зерна установить нельзя.\n"
        "texture: видимая форма, наклон и толщина слойков либо массивная текстура. Вертикальная ось "
        "керна не доказывает вертикальную слоистость; трещина не обязательно контакт пород.\n"
        "facies_name: гипотеза фации по наблюдаемым структурам, не просто название породы.\n"
        "facies_association: группа связанных фаций, не название породы.\n"
        "depositional_environment: гипотеза обстановки накопления по диагностическим признакам. "
        "Одна слоистость не доказывает русло. Если оснований для фациальной интерпретации мало, "
        "напиши 'Не установлена по фото'.\n"
        "Каждое классификационное поле — до 8 слов. Редкие деформационные текстуры называй только "
        "при различимых диагностических формах. Не придумывай минералы, глубины, координаты. "
        "Называй различимые обычные признаки конкретного образца, не повторяй шаблонные ответы.\n"
        "Контекст проекта: " + str(context)[:1000]
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
