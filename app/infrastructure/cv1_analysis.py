"""CV-1 contract: YOLO owns geometry; the local VLM supplies draft text only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES

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
    payload = [detection.label, detection.depth_from, detection.depth_to, detection.attributes]
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
    crop[mask == 0] = 255
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Не удалось подготовить фрагмент для локальной модели.")
    destination.write_bytes(encoded.tobytes())
    return destination


def apply_interval_description(record: PhotoRecord, job: IntervalJob, response: dict) -> None:
    """Apply validated strings, keeping YOLO pixels, depths and confidence intact."""
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
    attributes = dict(job.detection.attributes)
    attributes.pop("__expert_description_confirmed", None)
    attributes.pop("__expert_description_sha256", None)
    for key, names in TEXT_FIELDS.items():
        for name in names:
            attributes[name] = values[key]
    attributes.update({
        "__geometry_only": "1",
        "__geometry_sha256": job.geometry_sha256,
        "__geometry_record": record.identifier,
        "__interpretation_status": "draft",
        "Источник геометрии": "YOLO: полигон сегментации",
        "Источник описания": "локальная VLM; черновик",
        "Статус интерпретации": "черновик VLM, требуется проверка геолога",
    })
    model_info = response.get("model_info")
    if isinstance(model_info, dict):
        attributes["__vlm_model_info"] = json.dumps(model_info, ensure_ascii=False, sort_keys=True)
    job.detection.label = values["facies_name"] or UNRECOGNIZED_FACIES
    job.detection.attributes = attributes
    job.detection.training_ready = False


def mark_interval_failure(job: IntervalJob, message: str) -> None:
    """A failure is visible and never turns into a fabricated successful facies."""
    if not job.unchanged():
        return
    job.detection.attributes["__interpretation_status"] = "error"
    job.detection.attributes["Статус интерпретации"] = "описание не получено: " + str(message)[:600]
