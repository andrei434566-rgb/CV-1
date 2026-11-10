"""CV-1 contract: geometry is fixed; the local classifier supplies draft attributes.

Historical __vlm_* storage keys are retained for old project compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF

from app.domain.models import FaciesDetection, PhotoRecord
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.infrastructure.description_template import ASSUMPTIONS_KEY, SCHEMA, compose_description
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES
from app.infrastructure.core_description_prompt import check_interpretation_roles

PIPELINE_MODE = "yolo_vlm"
TEXT_FIELDS = {
    "facies_name": ("Название фации", "__vlm_facies_name"),
    "lithology": ("Название породы", "__vlm_lithology"),
    "texture": ("Текстура", "__vlm_texture"),
    "facies_association": ("Фациальная ассоциация", "__vlm_facies_association"),
    "depositional_environment": ("Обстановка седиментации", "__vlm_depositional_environment"),
    "description": ("Краткое описание", "__vlm_description"),
}


def geometry_signature(detection: FaciesDetection) -> str:
    coordinates = [[point.x(), point.y()] for point in detection.polygon]
    if len(coordinates) < 3 or not all(math.isfinite(value) for point in coordinates for value in point):
        raise ValueError("Контур интервала пуст или содержит некорректные координаты.")
    return hashlib.sha256(json.dumps(coordinates, allow_nan=False).encode("utf-8")).hexdigest()


def _input_signature(detection: FaciesDetection) -> str:
    # Reviewing current queue warnings does not edit geological input. A later
    # model result invalidates that review stamp through the queue fingerprint.
    attributes={key:value for key,value in detection.attributes.items() if key!='__review_queue_ack'}
    payload = [detection.label, detection.depth_from, detection.depth_to, attributes]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IntervalJob:
    record_id: str
    detection: FaciesDetection
    geometry_sha256: str
    input_sha256: str

    @classmethod
    def create(cls, record: PhotoRecord, detection: FaciesDetection) -> "IntervalJob":
        return cls(record.identifier, detection, geometry_signature(detection), _input_signature(detection))

    def unchanged(self) -> bool:
        try:
            return (
                not self.detection.training_ready
                and self.detection.attributes.get("__manual_contour") != "1"
                and self.detection.attributes.get("__manual_interval_edited") != "1"
                and geometry_signature(self.detection) == self.geometry_sha256
                and _input_signature(self.detection) == self.input_sha256
            )
        except (TypeError, ValueError):
            return False

    def is_current(self, record: PhotoRecord) -> bool:
        """Never overwrite a deleted, redrawn or expert-confirmed interval."""
        return (
            record.identifier == self.record_id
            and any(item is self.detection for item in record.detections)
            and self.unchanged()
        )


def create_core_column_description_detections(
    record: PhotoRecord,
    existing_detections: list[FaciesDetection] | None = None,
) -> list[FaciesDetection]:
    """Create neutral rows for detected columns with no facies polygons.

    These rectangles are useful for assembling and describing core photos, but
    they are explicitly marked as column-level geometry rather than facies
    masks so they cannot be mistaken for YOLO boundaries or training labels.
    """
    if record.pixmap.isNull() or not record.core_columns:
        return []
    from app.infrastructure.ml.core_column_service import normalize_columns

    columns = normalize_columns(
        record.core_columns,
        (record.pixmap.width(), record.pixmap.height()),
    )
    rows: list[FaciesDetection] = []
    existing = list(existing_detections or record.detections)
    represented_indices: set[int] = set()
    for detection in existing:
        if detection.attributes.get("__core_column_description") != "1":
            continue
        try:
            represented_indices.add(int(detection.attributes.get("__core_column_index", "")))
        except (TypeError, ValueError):
            continue
    for index, column in enumerate(columns):
        if index in represented_indices:
            continue
        column_width = max(1.0, column["right"] - column["left"])
        column_height = max(1.0, column["bottom"] - column["top"])
        already_has_mask = False
        for detection in existing:
            if detection.attributes.get("__core_column_description") == "1" or len(detection.polygon) < 3:
                continue
            xs = [point.x() for point in detection.polygon]
            ys = [point.y() for point in detection.polygon]
            overlap_x = max(0.0, min(max(xs), column["right"]) - max(min(xs), column["left"]))
            overlap_y = max(0.0, min(max(ys), column["bottom"]) - max(min(ys), column["top"]))
            if overlap_x >= column_width * 0.2 and overlap_y >= column_height * 0.08:
                already_has_mask = True
                break
        if already_has_mask:
            continue
        polygon = [
            QPointF(column["left"], column["top"]),
            QPointF(column["right"], column["top"]),
            QPointF(column["right"], column["bottom"]),
            QPointF(column["left"], column["bottom"]),
        ]
        segment = record.depth_segments[index] if index < len(record.depth_segments) else {}
        def optional_depth(key: str) -> float | None:
            try:
                value = float(segment[key])
            except (KeyError, TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        rows.append(FaciesDetection(
            label=UNRECOGNIZED_FACIES,
            confidence=0.0,
            polygon=polygon,
            attributes={
                "__geometry_only": "1",
                "__geometry_id": f"core-column:{record.identifier}:{index + 1}",
                "__geometry_provider": "core_column_detector",
                "__core_column_description": "1",
                "__core_column_index": str(index),
                "__interpretation_status": "waiting",
                "Источник геометрии": "Детектор столбиков керна: рамка целого столбика",
                "Статус интерпретации": "ожидает описания столбика локальной VLM",
            },
            depth_from=optional_depth("depth_from"),
            depth_to=optional_depth("depth_to"),
            training_ready=False,
        ))
    return rows


def write_interval_crop(record: PhotoRecord, job: IntervalJob, destination: Path) -> Path:
    """Map display coordinates to original pixels and hide everything outside the mask."""
    if not job.is_current(record):
        raise ValueError("Интервал изменён или подтверждён; автоматическое описание пропущено.")
    source = cv2.imdecode(np.frombuffer(Path(record.path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None or not source.size or record.pixmap.width() <= 0 or record.pixmap.height() <= 0:
        raise ValueError("Не удалось прочитать исходное фото интервала.")
    height, width = source.shape[:2]
    points = np.array([
        [point.x() * width / record.pixmap.width(), point.y() * height / record.pixmap.height()]
        for point in job.detection.polygon
    ], dtype=np.float64)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    contour = np.rint(points).astype(np.int32)
    if abs(cv2.contourArea(contour)) < 1:
        raise ValueError("Полигон не содержит различимого фрагмента исходного фото.")
    left, top, crop_width, crop_height = cv2.boundingRect(contour)
    crop = source[top:top + crop_height, left:left + crop_width].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour - np.array([left, top])], 255)
    if record.manual_exclusions:
        from app.infrastructure.photo_quality import manual_exclusion_mask
        excluded = manual_exclusion_mask(
            (width, height), record.manual_exclusions, (left, top, left + crop_width, top + crop_height))
        mask[excluded] = 0
    crop[mask == 0] = 255
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Alpha preserves the exact YOLO mask for photographic measurements; the
    # existing RGB pixels (including white exterior) remain unchanged for CNN.
    rgba=cv2.cvtColor(crop,cv2.COLOR_BGR2BGRA)
    rgba[:,:,3]=mask
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise ValueError("Не удалось подготовить фрагмент для локальной модели.")
    destination.write_bytes(encoded.tobytes())
    return destination


def apply_interval_description(record: PhotoRecord, job: IntervalJob, response: dict) -> None:
    """Apply validated strings while keeping detected geometry and depths intact."""
    if not job.is_current(record):
        raise ValueError("Интервал изменён во время ответа; результат VLM не применён.")
    fields = response.get("fields")
    if not response.get("ok"):
        raise ValueError(str(response.get("error") or "Локальная модель не ответила."))
    if response.get("incomplete"):
        raise ValueError("Описание оборвано; требуется повторный запрос или проверка геолога.")
    if not isinstance(fields, dict) or not isinstance(fields.get("description"), str) or not fields["description"].strip():
        raise ValueError("VLM не вернула структурированное описание интервала.")
    values: dict[str, str] = {}
    for key in TEXT_FIELDS:
        value = fields.get(key)
        if not isinstance(value, str):
            raise ValueError(f"Некорректный тип поля VLM: {key}.")
        value = value.strip()
        if len(value) > (6000 if key == "description" else 1000):
            raise ValueError(f"Слишком длинное поле VLM: {key}.")
        values[key] = value
    raw_values = dict(values)
    hydrodynamic = fields.get("hydrodynamic_regime")
    if hydrodynamic is not None and (not isinstance(hydrodynamic, str) or len(hydrodynamic) > 1000):
        raise ValueError("Некорректный гидродинамический режим VLM.")
    values, review_notes = check_interpretation_roles(values)
    attributes = dict(job.detection.attributes)
    previous_hydrodynamic = attributes.pop("__vlm_hydrodynamic_regime", None)
    if previous_hydrodynamic and attributes.get("Гидродинамический режим") == previous_hydrodynamic:
        attributes.pop("Гидродинамический режим", None)
    attributes.pop("__primary_reference", None)
    if hydrodynamic is not None:
        attributes["__vlm_hydrodynamic_regime"] = hydrodynamic.strip()
        attributes["Гидродинамический режим"] = hydrodynamic.strip()
    if isinstance(response.get("primary_reference"), dict):
        attributes["__primary_reference"] = json.dumps(response["primary_reference"], ensure_ascii=False)
    attributes.pop("__vlm_raw_fields", None)
    attributes.pop("__vlm_review_notes", None)
    if review_notes:
        attributes["__vlm_raw_fields"] = json.dumps(raw_values, ensure_ascii=False)
        attributes["__vlm_review_notes"] = json.dumps(review_notes, ensure_ascii=False)
    attributes.pop("__expert_description_confirmed", None)
    attributes.pop("__expert_description_sha256", None)
    for key, names in TEXT_FIELDS.items():
        for name in names:
            attributes[name] = values[key]
    is_core_column = attributes.get("__core_column_description") == "1"
    attributes.update({
        "__geometry_only": "1",
        "__geometry_sha256": job.geometry_sha256,
        "__geometry_record": record.identifier,
        "__interpretation_status": "draft",
        "Источник геометрии": (
            "Детектор столбиков керна: рамка целого столбика"
            if is_core_column else "YOLO masks.xy"
        ),
        "Источник описания": "локальный классификатор CV-1 Vision; описание по шаблону" if response.get("engine") == "vision" else "локальная VLM; черновик",
        "Статус интерпретации": "Автоматический черновик" if response.get("engine") == "vision" else "черновик VLM, требуется проверка геолога",
    })
    label = values["facies_name"] or UNRECOGNIZED_FACIES
    model_info = response.get("model_info")
    if isinstance(model_info, dict):
        attributes["__vlm_model_info"] = json.dumps(model_info, ensure_ascii=False, sort_keys=True)
    if response.get("engine") == "vision":
        attributes["__description_engine"] = "vision"
        attributes["__attribute_suggestions"] = json.dumps(response.get("attribute_suggestions", {}), ensure_ascii=False)
        attributes["__personal_predictions"] = json.dumps(response.get("personal_predictions", []), ensure_ascii=False)
        attributes["__vision_predictions"] = json.dumps(response.get("predictions", []), ensure_ascii=False)
        attributes["__vision_field_confidence"] = json.dumps(response.get("field_confidence", {}), ensure_ascii=False)
        attributes["__vision_facies_trained"] = "0"
        attributes["__vision_lithology_review_required"] = "1" if response.get("lithology_review_required", True) else "0"
        if isinstance(response.get("image_quality"),dict):
            attributes["__image_quality"] = json.dumps(response["image_quality"],ensure_ascii=False,allow_nan=False)
        observations = response.get("observations") or {}
        if isinstance(observations.get("visual_hint"), str):
            attributes["__vision_visual_hint"] = observations["visual_hint"][:1000]
        color = observations.get("color")
        from app.infrastructure.core_color_hint import apply_photo_color,color_is_confirmed
        previous_auto_color=attributes.get('__vision_color')
        auto_color_matches=bool(previous_auto_color and attributes.get('Цвет')==previous_auto_color)
        if isinstance(color,str) and color:
            measurement=observations.get('color_measurement')
            apply_photo_color(attributes,measurement if isinstance(measurement,dict) else {'label':color})
        elif 'color' in observations and not color and auto_color_matches and not color_is_confirmed(attributes):
            attributes.pop('Цвет',None);attributes.pop('__vision_color',None);attributes.pop('__color_measurement',None)
        draft_observations = response.get("draft_observations")
        if isinstance(draft_observations, dict):
            for field in LITHOLOGY_ATTRIBUTE_OPTIONS:
                value = draft_observations.get(field)
                if isinstance(value, str) and value.strip() and (field != "Цвет" or not attributes.get(field)):
                    attributes[field] = value.strip()[:1000]
            assumptions = {field for field in response.get("draft_assumptions", ()) if isinstance(field, str)}
            if color_is_confirmed(attributes):
                assumptions.discard("Цвет")
            reference_label = response.get("facies_label")
            if isinstance(reference_label, str) and reference_label and values["facies_name"] == raw_values["facies_name"]:
                label = reference_label
                attributes["Код фации"] = label
                if response.get("facies_index"):
                    attributes["Индекс фации"] = str(response["facies_index"])
            attributes[ASSUMPTIONS_KEY] = json.dumps(sorted(assumptions), ensure_ascii=False)
            draft = compose_description(attributes, job.detection.depth_from, job.detection.depth_to,
                                        facies_name=values["facies_name"], assumptions=assumptions)
            attributes[ASSUMPTIONS_KEY] = json.dumps(draft.assumed, ensure_ascii=False)
            attributes["__description_mode"] = "template"
            attributes["__description_composer"] = SCHEMA
            attributes["__description_template_text"] = draft.text
            for name in TEXT_FIELDS["description"]:
                attributes[name] = draft.text
    if isinstance(response.get("timing"), dict):
        attributes["__vlm_timing"] = json.dumps(response["timing"], ensure_ascii=False, sort_keys=True)
    job.detection.label = label
    job.detection.attributes = attributes
    job.detection.training_ready = False


def mark_interval_failure(job: IntervalJob, message: str) -> None:
    """A failure is visible and never turns into a fabricated successful facies."""
    if not job.unchanged():
        return
    job.detection.attributes["__interpretation_status"] = "error"
    job.detection.attributes["Статус интерпретации"] = "описание не получено: " + str(message)[:600]
