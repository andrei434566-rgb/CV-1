"""Preserve detail in a very wide, single-row core photo for VLM inference.

This only changes the model input image. It never assigns depth to pixels.
"""

from __future__ import annotations

import math

from PIL import Image, ImageOps


def prepare(image: Image.Image, max_side: int = 896, max_tiles: int = 4) -> tuple[Image.Image, int]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width / height < 3:
        image.thumbnail((max_side, max_side))
        return image, 1
    count = min(max_tiles, max(2, math.ceil(width / (2 * height))))
    panels = []
    for index in range(count):
        left = round(index * width / count)
        right = round((index + 1) * width / count)
        panels.append(image.crop((left, 0, right, height)))
    panel_width = max(panel.width for panel in panels)
    mosaic = Image.new("RGB", (panel_width, height * count), "white")
    for index, panel in enumerate(panels):
        mosaic.paste(panel, (0, index * height))
    mosaic.thumbnail((max_side, max_side))
    return mosaic, count
