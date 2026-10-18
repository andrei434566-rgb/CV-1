"""Make source-linked review windows from three horizontal Fire Island photos.

Depth increases right-to-left on each ruler; photos are rotated 90 degrees
counterclockwise for the VLM so shallower material appears above deeper.
These are experimental candidates, not automatically approved labels.
"""

from __future__ import annotations

import hashlib
import html
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_fire_island_2016"
OUT = Path(__file__).resolve().parent / "data/usgs_fire_island_horizontal_pilot"
# Endpoints were read against the printed 0/100/200 cm rulers. Windows below
# stay at least ~2 cm inside source interval boundaries to absorb sub-cm lens
# distortion and reading error. Coordinates are on the original JPEGs.
GEOMETRY = {
    "P10-C03": {"rows": [{"depth_cm": [0, 100], "ruler_x_px": [3778, 40],
                           "core_y_px": [555, 820], "ruler_and_core_y_px": [444, 830]}]},
    "P25-C06": {"rows": [{"depth_cm": [0, 100], "ruler_x_px": [3764, 44],
                           "core_y_px": [545, 787], "ruler_and_core_y_px": [415, 800]},
                          {"depth_cm": [100, 200], "ruler_x_px": [3760, 40],
                           "core_y_px": [1025, 1275], "ruler_and_core_y_px": [925, 1290]}]},
    "P29-C11": {"rows": [{"depth_cm": [0, 100], "ruler_x_px": [3756, 32],
                           "core_y_px": [560, 820], "ruler_and_core_y_px": [448, 835]},
                          {"depth_cm": [100, 200], "ruler_x_px": [3750, 32],
                           "core_y_px": [1050, 1315], "ruler_and_core_y_px": [935, 1330]}]},
}
WINDOWS = [
    ("P10-C03", 3, 14, "Светлый пятнистый песок без отчётливой слоистости."),
    ("P10-C03", 20, 26, "Тёмный песок с изогнутыми светлыми полосами."),
    ("P10-C03", 42, 55, "Песок с контрастными тёмными и светлыми деформированными полосами."),
    ("P10-C03", 62, 85, "Светлый песок без отчётливой слоистости."),
    ("P25-C06", 2, 12, "Песок со светлыми и тёмными участками без чёткой слоистости."),
    ("P25-C06", 20, 39, "Песок с неравномерными тёмными пятнистыми участками."),
    ("P25-C06", 52, 82, "Светлый песок без отчётливой слоистости."),
    ("P25-C06", 124, 138, "Светлый песок с тёмными волнистыми прослоями."),
    ("P25-C06", 140, 155, "Светлый песок с тёмными изогнутыми прослоями."),
    ("P29-C11", 2, 16, "Тёмный песок с неравномерной окраской."),
    ("P29-C11", 22, 37, "Светлый песок без отчётливой слоистости."),
    ("P29-C11", 44, 58, "Песок с мощным тёмным деформированным прослоем."),
    ("P29-C11", 72, 96, "Светлый песок с тонкими тёмными изогнутыми прослоями."),
    ("P29-C11", 123, 138, "Светлый песок без отчётливой слоистости."),
]


def row_for(core_id: str, top: float, base: float) -> dict:
    rows = [row for row in GEOMETRY[core_id]["rows"]
            if row["depth_cm"][0] <= top < base <= row["depth_cm"][1]]
    if len(rows) != 1:
        raise ValueError(f"Window crosses or misses ruler rows: {core_id} {top}-{base}")
    return rows[0]


def depth_to_x(row: dict, depth_cm: float) -> int:
    d0, d1 = row["depth_cm"]
    x0, x1 = row["ruler_x_px"]
    return round(x0 + (depth_cm - d0) / (d1 - d0) * (x1 - x0))


def main() -> None:
    source = json.loads((ROOT / "source_intervals.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    records = []
    with ZipFile(ROOT / "2016-322-FA_CorePhotos.zip") as archive:
        for core_id, top, base, draft in WINDOWS:
            parents = [r for r in source["intervals"] if r["core_id"] == core_id
                       and r["source_interval_cm"][0] <= top
                       and base <= r["source_interval_cm"][1]]
            if len(parents) != 1:
                raise ValueError(f"Expected one source interval: {core_id} {top}-{base}")
            parent = parents[0]
            row = row_for(core_id, top, base)
            x0, x1 = depth_to_x(row, base), depth_to_x(row, top)
            y0, y1 = row["core_y_px"]
            member = f"2016-322-FA_{core_id}.jpg"
            raw = archive.read(member)
            image = Image.open(BytesIO(raw)).convert("RGB")
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ValueError(f"Crop outside original image: {core_id} {top}-{base}")
            stem = f"{core_id}_{top:g}-{base:g}cm"
            core_path = OUT / "images" / f"{stem}_core.jpg"
            ruler_path = OUT / "images" / f"{stem}_ruler.jpg"
            # CCW turns right-to-left depth progression into top-to-bottom.
            image.crop((x0, y0, x1, y1)).rotate(90, expand=True).save(core_path, quality=95)
            ry0, ry1 = row["ruler_and_core_y_px"]
            image.crop((x0, ry0, x1, ry1)).rotate(90, expand=True).save(ruler_path, quality=92)
            records.append({
                "id": f"fire_island_2016_{stem}", "core_id": core_id,
                "crop_interval_cm": [top, base],
                "source_interval_cm": parent["source_interval_cm"],
                "source_record_id": parent["record_id"],
                "source_pdf_page": parent["source_pdf_page"],
                "source_description_en": parent["source_description_en"],
                "visible_target_ru_draft": draft,
                "source_photo_member": member,
                "source_photo_sha256": hashlib.sha256(raw).hexdigest(),
                "crop_box_xyxy_original_px": [x0, y0, x1, y1],
                "image_size_original_px": list(image.size),
                "core_crop": str(core_path.relative_to(OUT)).replace("\\", "/"),
                "core_crop_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
                "ruler_crop": str(ruler_path.relative_to(OUT)).replace("\\", "/"),
                "alignment_method": "manual 0/100/200 cm ruler endpoints; affine interpolation; source windows kept inside boundaries",
                "rotation_ccw_deg": 90,
                "ruler_row_depth_cm": row["depth_cm"],
                "photo_interval_aligned": True,
                "russian_visual_target_reviewed": False,
                "training_approved": False,
            })
    manifest = {
        "source_url": source["source_url"],
        "source_pdf_sha256": source["source_pdf_sha256"],
        "license": "CC0 1.0 Universal",
        "geometry": GEOMETRY,
        "candidates": records,
        "candidate_count": len(records), "training_approved_count": 0,
        "warning": "Window labels require individual visual review; these are not geologist-approved production targets.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for r in records:
        cards.append("<article><h2>" + html.escape(r["core_id"])
                     + f" · {r['crop_interval_cm'][0]:g}–{r['crop_interval_cm'][1]:g} см"
                     + "</h2><div class='images'><img src='" + html.escape(r["core_crop"])
                     + "'><img src='" + html.escape(r["ruler_crop"])
                     + "'></div><p><b>Черновик:</b> " + html.escape(r["visible_target_ru_draft"])
                     + "</p><p><b>Исходный журнал:</b> " + html.escape(r["source_description_en"])
                     + "</p><p>Источник: " + html.escape(r["source_photo_member"])
                     + f", PDF с. {r['source_pdf_page']}, пиксели: "
                     + html.escape(str(r["crop_box_xyxy_original_px"])) + "</p></article>")
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>Fire Island: горизонтальные окна</title>
<style>body{font:16px system-ui;margin:2rem auto;max-width:1500px;background:#f5f7fa;color:#223}article{background:white;margin:1.5rem 0;padding:1.5rem;border:1px solid #ccd;border-radius:12px}.images{display:flex;gap:1rem;align-items:flex-start;overflow:auto}.images img{max-width:47%;max-height:850px;object-fit:contain}p{line-height:1.5}</style>
<h1>USGS Fire Island: горизонтальные фото, интервалы по линейке</h1><p>Фото повёрнуты так, что глубина идёт сверху вниз. Слева керн, справа керн с линейкой. Русский текст — черновой; ни один кандидат не допущен к производственному обучению.</p>""" + "".join(cards) + "</html>"
    (OUT / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"candidate_count": len(records), "cores": len({r['core_id'] for r in records}),
                      "training_approved": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
