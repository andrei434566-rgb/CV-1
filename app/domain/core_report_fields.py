"""Expert-entered fields shared by the layer card, report and reviewed examples."""

INTERVAL_SHIFT_FIELD = "Смещение внутри интервала, м"
REPORT_CHOICE_FIELDS = (
    "Индекс макрофации", "Название макрофации", "Код макрофации",
    "Индекс ассоциации фаций", "Комплекс осадконакопления",
)
REPORT_NOTE_FIELD = "Примечания"
LAYER_REPORT_FIELDS = (INTERVAL_SHIFT_FIELD, *REPORT_CHOICE_FIELDS, REPORT_NOTE_FIELD)

_ALIASES = {
    INTERVAL_SHIFT_FIELD: ("Смещение внутри интервала",),
    "Индекс ассоциации фаций": ("Индексс ассоциации фаций",),
    REPORT_NOTE_FIELD: ("Примечание",),
}


def report_field_value(attributes: dict, field: str) -> str:
    for key in (field, *_ALIASES.get(field, ())):
        if key in attributes:
            value = attributes[key]
            return "" if value is None else str(value).strip()
    return ""


def layer_report_fields(attributes: dict) -> dict[str, str]:
    return {field: value for field in LAYER_REPORT_FIELDS if (value := report_field_value(attributes, field))}
