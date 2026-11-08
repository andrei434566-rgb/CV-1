"""Independent visual attribute suggester for already segmented facies."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QPolygonF

from app.domain.models import FaciesDetection, PhotoRecord


class LithologyAttributeService:
    """Suggest observable attributes without changing the facies segmentation.

    Facies boundaries remain the responsibility of YOLO.  This service works
    only inside each accepted mask. Colour is measured from original pixels;
    untrained grain-size or bedding guesses are not added from image contrast.
    """

    def suggest(self, record: PhotoRecord, detection: FaciesDetection) -> dict[str, str]:
        if detection.attributes.get("__description_engine") == "vision":
            # Laplacian contrast is not a calibrated grain-size measurement.
            # Never silently add untrained grain/bedding labels to CNN results.
            import json
            try:measurement=json.loads(detection.attributes.get('__color_measurement','{}'))
            except (TypeError,ValueError):measurement={}
            color=detection.attributes.get('__vision_color')
            if isinstance(measurement,dict) and measurement.get('schema')=='cv1-core-color-v2' and color:
                return {'Цвет':color}
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(Path(record.path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or record.pixmap.isNull():
                return {}
            rect = QPolygonF(detection.polygon).boundingRect()
            x_scale = image.shape[1] / max(1, record.pixmap.width())
            y_scale = image.shape[0] / max(1, record.pixmap.height())
            left = max(0, int(rect.left() * x_scale))
            top = max(0, int(rect.top() * y_scale))
            right = min(image.shape[1], int(rect.right() * x_scale))
            bottom = min(image.shape[0], int(rect.bottom() * y_scale))
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                return {}
            mask=np.zeros(crop.shape[:2],dtype='uint8')
            vertices=np.array([[round(p.x()*x_scale-left),round(p.y()*y_scale-top)] for p in detection.polygon],dtype='int32')
            if len(vertices)<3:return {}
            cv2.fillPoly(mask,[vertices],255)
            return self._suggest_crop(crop,mask)
        except (ImportError, OSError):
            return {}

    @staticmethod
    def _suggest_crop(crop,mask=None) -> dict[str,str]:
        from PIL import Image
        from app.infrastructure.core_color_hint import estimate_core_color
        from app.infrastructure.photo_quality import label_boxes
        image=Image.fromarray(crop[:,:,::-1].copy())
        hint=estimate_core_color(image,mask,exclude_boxes=label_boxes(image,valid_mask=mask))
        return {'Цвет':hint.label} if hint else {}
