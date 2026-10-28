"""Conservative photographic colour hints for a cropped core interval.

This is not a calibrated colour measurement or a facies classifier.  A label
is offered only when most usable pixels in the inner interval agree.  The
caller must pass original photo pixels, not a coloured overlay or a rendered
geological column, and should present the result as a suggestion to review.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class CoreColorHint:
    label: str
    supporting_fraction: float
    usable_pixels: int
    limitation: str = "Цвет по фото ориентировочный: зависит от освещения, баланса белого и состояния керна; чёрно-белое фото не передаёт истинный цвет."


def estimate_core_color(image: Image.Image, mask: Image.Image | np.ndarray | None = None) -> CoreColorHint | None:
    """Suggest grey/brown only when one clearly dominates original photo pixels.

    Ignore the crop's narrow perimeter (often tray edges) and near-white or
    almost-black pixels (background, labels and extreme shadows).  An optional
    binary mask excludes pixels outside the recognized core.  If too little
    remains or colours are mixed, no hint is returned; this method cannot
    reliably identify labels that have a similar colour to the core itself.
    """
    if not isinstance(image, Image.Image):
        return None
    # Keep the pixel coordinate system unchanged: a supplied mask must align.
    # Callers that apply EXIF orientation must apply it to image and mask alike.
    image = image.convert("RGB")
    width, height = image.size
    if height < 12 or width < 12:
        return None
    if mask is None:
        selected = np.ones((height, width), dtype=bool)
    else:
        try:
            mask_array = np.asarray(mask)
        except (ValueError, TypeError):
            return None
        if mask_array.shape != (height, width):
            return None
        selected = mask_array.astype(bool, copy=True)
    # Ignore edges, where detected boxes often include trays or mask outlines.
    x_margin, y_margin = max(1, round(width * 0.12)), max(1, round(height * 0.03))
    selected[:, :x_margin] = False
    selected[:, width - x_margin:] = False
    selected[:y_margin, :] = False
    selected[height - y_margin:, :] = False
    if selected.sum() < 64:
        return None
    hsv = np.asarray(image.convert("HSV"))
    h, s, v = (hsv[:, :, i] for i in range(3))
    # Near-white labels/background and black shadows are not reliable rock colour.
    usable = selected & (v >= 20) & (v < 238)
    count = int(usable.sum())
    if count < 64 or count / int(selected.sum()) < 0.30:
        return None
    grey = usable & (s <= 35)
    brown = usable & (s >= 45) & (h >= 6) & (h <= 40)
    grey_fraction = int(grey.sum()) / count
    brown_fraction = int(brown.sum()) / count
    if grey_fraction >= 0.75:
        brightness = float(np.median(v[grey]))
        label = "тёмно-серый" if brightness < 95 else "серый" if brightness < 165 else "светло-серый"
        return CoreColorHint(label, grey_fraction, count)
    if brown_fraction >= 0.75:
        return CoreColorHint("коричневый", brown_fraction, count)
    return None


def estimate_core_color_file(path: str | Path, mask: Image.Image | np.ndarray | None = None) -> CoreColorHint | None:
    """Read a crop without assuming an ASCII filesystem path."""
    try:
        with Image.open(Path(path)) as image:
            return estimate_core_color(image, mask)
    except (OSError, ValueError, UnidentifiedImageError):
        return None
