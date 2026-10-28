"""Expert-confirmed facies linked to a specific YOLO observation.

Mappings live in project detection attributes. They are local context for the
same photo/contour only, not a global class translation or training a model.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MAPPING_KEY = "__expert_facies_mapping"


def source_checkpoint_identity(path: str | Path) -> dict[str, str]:
    """Identify actual weights, never treating a name/path as their identity."""
    path = Path(path)
    name = path.name
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"name": name, "sha256": digest.hexdigest()}
    except OSError:
        return {"name": name, "sha256": ""}


def _photo_sha256(record) -> str:
    digest = hashlib.sha256()
    with Path(record.path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(record, detection) -> dict:
    width, height = record.pixmap.width(), record.pixmap.height()
    if width <= 0 or height <= 0 or len(detection.polygon) < 3:
        raise ValueError("Нужны исходное фото и полигон фации.")
    points = []
    for point in detection.polygon:
        x, y = float(point.x()), float(point.y())
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Полигон содержит некорректные координаты.")
        points.append([round(x / width, 6), round(y / height, 6)])
    values = detection.attributes or {}
    label = str(detection.label or "").strip()
    if not label or label in {"Интервал YOLO", "Новый контур"}:
        raise ValueError("Нужна выбранная специалистом фация.")
    depths = []
    for value in (detection.depth_from, detection.depth_to):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("Некорректная глубина.")
        depths.append(None if value is None else float(value))
    return {
        "photo_sha256": _photo_sha256(record),
        "record_id": str(record.identifier),
        "well": str(record.well_name or ""),
        "polygon_relative": points,
        "depth_from": depths[0], "depth_to": depths[1],
        "expert_facies_label": label,
        "expert_facies_index": str(values.get("Индекс фации") or "").strip(),
        "expert_facies_name": str(values.get("Название фации") or "").strip(),
        "source_class": str(values.get("__source_model_class") or "").strip(),
        "source_class_id": str(values.get("__source_model_class_id") or "").strip(),
        "model_name": str(values.get("__source_model_name") or "").strip(),
        "model_sha256": str(values.get("__source_model_sha256") or "").strip(),
    }


def save_expert_facies_mapping(record, detection, *, confirmed: bool) -> dict | None:
    """Record a manual confirmation; never replace the original YOLO class.

    ``confirmed`` means the specialist committed this facies for this interval;
    ``training_ready`` remains a separate explicit training permission.
    An unconfirmed edit invalidates any older association immediately.
    """
    if not confirmed:
        detection.attributes.pop(MAPPING_KEY, None)
        return None
    state = _state(record, detection)
    row = {"schema": "cv1-expert-facies-mapping-v1", "confirmed": True,
           "training_ready": bool(detection.training_ready), **state}
    detection.attributes[MAPPING_KEY] = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return row


def current_expert_facies_mapping(record, detection) -> dict | None:
    """Return a confirmation only for unchanged photo, contour, facies and YOLO provenance."""
    try:
        row = json.loads((detection.attributes or {}).get(MAPPING_KEY, ""))
        if not isinstance(row, dict) or row.get("schema") != "cv1-expert-facies-mapping-v1" or row.get("confirmed") is not True:
            return None
        state = _state(record, detection)
        if any(row.get(key) != value for key, value in state.items()):
            return None
        if row.get("training_ready") is not bool(detection.training_ready):
            return None
        return row
    except (OSError, ValueError, TypeError, AttributeError, OverflowError):
        return None


def facies_mapping_context(record, detection) -> str:
    """A VLM may use the expert value for this exact interval, not other images."""
    row = current_expert_facies_mapping(record, detection)
    if row is None:
        return ""
    facies = row["expert_facies_label"][:120]
    index = row["expert_facies_index"][:40]
    name = row["expert_facies_name"][:240]
    text = "Фация, сохранённая специалистом для этого интервала: " + facies
    if index:
        text += "; индекс: " + index
    if name and name != facies:
        text += "; название: " + name
    if row["source_class"]:
        text += ". Исходная метка YOLO: " + row["source_class"][:120]
        if row["source_class_id"]:
            text += " (id " + row["source_class_id"][:20] + ")"
        if row["model_sha256"]:
            text += "; веса модели идентифицированы SHA256 " + row["model_sha256"]
        else:
            text += "; идентичность файла весов не подтверждена"
        text += ". Метка YOLO может обозначать геометрию и сама по себе не доказывает фацию."
    else:
        text += ". Для этого интервала нет исходной метки YOLO."
    return text + " Данные относятся только к текущему контуру; не переносите их на другие интервалы."
