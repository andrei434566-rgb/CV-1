"""Create a provenance-first review index for 2016 Fire Island core pairs."""

from __future__ import annotations

import html
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_fire_island_2016"
PREVIEWS = ROOT / "review_previews"
PHOTO = ROOT / "2016-322-FA_CorePhotos.zip"
LOG = ROOT / "2016-322-FA_CoreLogs.zip"
SOURCE = "https://www.usgs.gov/data/archive-sediment-data-vibracores-collected-2016-fire-island-new-york"
LOG_FILENAME_EXCEPTION = {"P29-C10": "2016-322-FA_CoreLog-P29-C09.jpg"}


def preview(archive: zipfile.ZipFile, name: str, prefix: str) -> tuple[str, list[int]]:
    destination = PREVIEWS / f"{prefix}_{name}"
    with Image.open(io.BytesIO(archive.read(name))) as opened:
        original_size = list(opened.size)
        if not destination.exists():
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((1800, 1800))
            image.save(destination, quality=87, optimize=True)
    return destination.relative_to(ROOT).as_posix(), original_size


def main() -> None:
    PREVIEWS.mkdir(exist_ok=True)
    records = {}
    cards = []
    with zipfile.ZipFile(PHOTO) as photos, zipfile.ZipFile(LOG) as logs:
        ids = sorted(re.fullmatch(r"2016-322-FA_(P\d+-C\d+)\.jpg", name).group(1)
                     for name in photos.namelist()
                     if re.fullmatch(r"2016-322-FA_(P\d+-C\d+)\.jpg", name))
        if len(ids) != 14:
            raise ValueError(f"Expected 14 cores, found {len(ids)}")
        for core_id in ids:
            photo_name = f"2016-322-FA_{core_id}.jpg"
            log_name = LOG_FILENAME_EXCEPTION.get(core_id, f"2016-322-FA_CoreLog-{core_id}.jpg")
            if log_name not in logs.namelist():
                raise ValueError(f"Missing log for {core_id}: {log_name}")
            photo_file, photo_size = preview(photos, photo_name, "photo")
            log_file, log_size = preview(logs, log_name, "log")
            anomaly = ("Имя файла журнала указывает P29-C09, но его напечатанный Core ID — P29-C10; "
                       "сверено вручную с фото P29-C10. "
                       if core_id == "P29-C10" else "")
            records[core_id] = {"photo_member": photo_name, "log_member": log_name,
                                "photo_pixels": photo_size, "log_pixels": log_size,
                                "same_core_id_in_filename": core_id != "P29-C10",
                                "log_internal_id_visually_checked": core_id == "P29-C10",
                                "filename_anomaly_note": anomaly or None,
                                "depth_aligned": False, "training_approved": False}
            cards.append(f"<details><summary><b>{html.escape(core_id)}</b>"
                         f"{' ⚠ имя журнала' if anomaly else ''}</summary>"
                         f"<p>{html.escape(anomaly)}</p><div class='pair'>"
                         f"<figure><a href='{photo_file}'><img loading='lazy' src='{photo_file}'></a>"
                         f"<figcaption>Фото {photo_size[0]}×{photo_size[1]}</figcaption></figure>"
                         f"<figure><a href='{log_file}'><img loading='lazy' src='{log_file}'></a>"
                         f"<figcaption>Журнал {log_size[0]}×{log_size[1]}</figcaption></figure>"
                         f"</div></details>")
    report = {"source_url": SOURCE, "doi": "10.5066/F7FN15GX",
              "license": "CC0 1.0 Universal on USGS data release",
              "core_pairs": records, "pairs_by_core_id": len(records),
              "filename_exception_count": len(LOG_FILENAME_EXCEPTION),
              "interval_pairs_training_approved": 0}
    (ROOT / "core_pair_index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS Fire Island 2016</title>
<style>body{font:16px system-ui;max-width:1600px;margin:2rem auto;background:#f7f9fb;color:#20313d}details{background:white;border:1px solid #ccd;border-radius:.6rem;padding:1rem;margin:1rem 0}.pair{display:flex;gap:1rem;overflow:auto}figure{margin:0;min-width:45%}img{max-width:100%;max-height:900px}figcaption{font-size:.85rem}</style>
<h1>USGS Fire Island 2016 — 14 кернов</h1><p>Официальные архивы фото и журналов, DOI 10.5066/F7FN15GX, CC0. Сопоставление по ID — только первый шаг. Границы интервалов ещё не привязаны к фото, утверждённых примеров для обучения: 0.</p>""" + "".join(cards) + "</html>"
    (ROOT / "review_index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"paired_cores": len(records), "previews": 2 * len(records),
                      "filename_exceptions": len(LOG_FILENAME_EXCEPTION),
                      "training_approved": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
