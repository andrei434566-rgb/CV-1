"""Build a seven-core photo/log review sheet without creating training labels."""

from __future__ import annotations

import html
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_midnight_pass"
PREVIEWS = ROOT / "review_previews"
PHOTO = ROOT / "2024-330-FA_MP_CorePhotos.zip"
LOG = ROOT / "2024-330-FA_MP_CoreLogs.zip"
IDS = ("MP2-D", "MP2-E", "MP2-F", "MP2-G", "MP3-B", "MP3-D", "MP3-E")


def make_preview(archive: zipfile.ZipFile, member: str, prefix: str) -> tuple[str, list[int]]:
    target = PREVIEWS / f"{prefix}_{member}"
    with archive.open(member) as stream:
        with Image.open(io.BytesIO(stream.read())) as source:
            size = list(source.size)
            if not target.exists():
                picture = ImageOps.exif_transpose(source).convert("RGB")
                picture.thumbnail((2400, 2400))
                picture.save(target, quality=88, optimize=True)
    return target.relative_to(ROOT).as_posix(), size


def main() -> None:
    PREVIEWS.mkdir(exist_ok=True)
    cards = []
    core_index = {}
    with zipfile.ZipFile(PHOTO) as photos, zipfile.ZipFile(LOG) as logs:
        for core_id in IDS:
            photo_name = f"{core_id}.jpg"
            log_name = f"2024-330-FA-{core_id}.jpg"
            if photo_name not in photos.namelist() or log_name not in logs.namelist():
                raise ValueError(f"Missing official photo/log pair: {core_id}")
            photo_preview, photo_size = make_preview(photos, photo_name, "photo")
            log_preview, log_size = make_preview(logs, log_name, "log")
            core_index[core_id] = {"photo_member": photo_name, "log_member": log_name,
                                   "photo_pixels": photo_size, "log_pixels": log_size,
                                   "id_match": True, "depth_aligned": False,
                                   "training_approved": False}
            cards.append(f"<details><summary><b>{html.escape(core_id)}</b> — фото и журнал</summary>"
                         f"<div class='strip'><figure><a href='{photo_preview}'><img src='{photo_preview}'></a>"
                         f"<figcaption>Фото {photo_size[0]}×{photo_size[1]} пикс.</figcaption></figure>"
                         f"<figure><a href='{log_preview}'><img src='{log_preview}'></a>"
                         f"<figcaption>Журнал {log_size[0]}×{log_size[1]} пикс.</figcaption></figure>"
                         f"</div></details>")
    (ROOT / "core_pair_index.json").write_text(json.dumps({"source_url":
        "https://www.usgs.gov/data/sediment-data-cores-collected-2025-midnight-pass-florida",
        "license": "CC0 1.0 Universal", "cores": core_index,
        "training_approved": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    document = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS Midnight Pass</title>
<style>body{font:16px sans-serif;max-width:1500px;margin:2rem auto;background:#f7f9fb;color:#20313d}details{background:#fff;border:1px solid #ccd;border-radius:.5rem;margin:1rem 0;padding:1rem}summary{cursor:pointer}.strip{display:flex;gap:1rem;overflow:auto}figure{margin:0;min-width:300px}img{max-width:680px;max-height:720px}figcaption{font-size:.85rem}</style>
<h1>USGS Midnight Pass: проверка фото и журналов</h1><p><a href='https://www.usgs.gov/data/sediment-data-cores-collected-2025-midnight-pass-florida'>Официальный источник</a>, DOI 10.5066/P1CQ6QRR, CC0. Совпадение ID подтверждено. Перед обучением нужно связать глубину каждого участка снимка с интервалом журнала и сверить видимые признаки. Сейчас утверждённых фотоописаний нет.</p>""" + "".join(cards) + "</html>"
    (ROOT / "review_index.html").write_text(document, encoding="utf-8")
    print(json.dumps({"paired_cores": len(core_index), "previews": len(list(PREVIEWS.glob("*.jpg")))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
