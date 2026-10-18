"""Print image-pixel positions of labelled 10 cm ruler ticks for vertical photos.

This is an inspection aid only. Tick candidates need visual confirmation before
being used to crop any photo-description interval.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_fire_island_2016"
SPECS = {
    "P22-C14": {"ruler_x": (465, 510), "y0_guess": 145, "pixels_per_cm_guess": 35.6},
    "P25-C05": {"ruler_x": (1110, 1170), "y0_guess": 309, "pixels_per_cm_guess": 34.5},
}


def main() -> None:
    with ZipFile(ROOT / "2016-322-FA_CorePhotos.zip") as photos:
        for core_id, spec in SPECS.items():
            name = f"2016-322-FA_{core_id}.jpg"
            image = np.asarray(Image.open(BytesIO(photos.read(name))).convert("L"))
            x0, x1 = spec["ruler_x"]
            darkness = image[:, x0:x1].mean(axis=1)
            print(core_id, image.shape[::-1])
            for cm in range(0, 101, 10):
                predicted = spec["y0_guess"] + spec["pixels_per_cm_guess"] * cm
                lo = max(0, round(predicted - 35))
                hi = min(len(darkness), round(predicted + 35))
                candidates = np.argsort(darkness[lo:hi])[:8] + lo
                ranked = [(int(y), round(float(darkness[y]), 1)) for y in candidates]
                print(f"  {cm:3d} cm, predicted {predicted:.1f}px: {ranked}")


if __name__ == "__main__":
    main()
