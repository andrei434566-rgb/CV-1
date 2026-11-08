"""Find physical core columns in a chat attachment without blocking Qt UI."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Signal, Slot

from app.infrastructure.ml.core_column_service import CoreColumnRecognizer, read_bgr_image


class ChatPhotoColumnWorker(QObject):
    succeeded = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, image_path: Path, model_path: Path | None = None):
        super().__init__()
        self.image_path = Path(image_path)
        self.model_path = Path(model_path) if model_path else None

    @Slot()
    def run(self) -> None:
        try:
            image = read_bgr_image(self.image_path)
            height, width = image.shape[:2]
            if width < 16 or height < 16:
                raise ValueError("Изображение слишком маленькое для поиска столбиков.")
            max_side = max(width, height)
            scale = min(1.0, 6000.0 / max_side, math.sqrt(8_000_000 / (width * height)))
            working = cv2.resize(
                image,
                (max(16, round(width * scale)), max(16, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            ) if scale < 1.0 else image
            recognizer = CoreColumnRecognizer(self.model_path)
            columns = recognizer.recognize(working, target_size=(width, height))
            detector = recognizer.source_label
            if not columns and height >= width * 3.2:
                # A long single-column core strip can fill nearly the whole
                # frame, leaving no inter-column gaps for the tray detector.
                # Mark this explicitly as a tentative geometry fallback.
                columns = [{
                    "left": width * 0.02, "top": height * 0.01,
                    "right": width * 0.98, "bottom": height * 0.99,
                }]
                detector = "геометрическое предположение: один вытянутый столбик"
            from app.infrastructure.photo_quality import audit_image
            from PIL import Image
            quality=audit_image(Image.fromarray(working[:,:,::-1]),
                [{k:v*(working.shape[1]/width if k in {'left','right'} else working.shape[0]/height) for k,v in box.items()} for box in columns],
                source_size=(width,height))
            self.succeeded.emit({
                "path": str(self.image_path),
                "width": width,
                "height": height,
                "columns": columns,
                "detector": detector,
                "quality": quality,
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()
