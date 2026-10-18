"""Make a local core-by-core visual review of CC0 Florida Keys photos and logs."""

from __future__ import annotations

import html
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).parent / "data" / "external_raw" / "usgs_florida_keys_reef"
PREVIEWS = ROOT / "review_previews"
INDEX = ROOT / "inventory_and_core_index.json"
PHOTO_ZIP = ROOT / "FKRT_Core_Photographs.zip"
LOG_ZIP = ROOT / "FKRT_Descriptive_Core_Logs.zip"


def preview(archive: zipfile.ZipFile, member: str, kind: str) -> str:
    name = (kind + "_" + member.replace("/", "_").replace("\\", "_")).rsplit(".", 1)[0] + ".jpg"
    target = PREVIEWS / name
    if not target.exists():
        with archive.open(member) as stream:
            image = Image.open(io.BytesIO(stream.read())).convert("RGB")
            image = ImageOps.exif_transpose(image)
            image.thumbnail((1300, 1300))
            image.save(target, quality=84, optimize=True)
    return target.relative_to(ROOT).as_posix()


def main() -> None:
    PREVIEWS.mkdir(exist_ok=True)
    report = json.loads(INDEX.read_text(encoding="utf-8"))
    cards = []
    with zipfile.ZipFile(PHOTO_ZIP) as photos, zipfile.ZipFile(LOG_ZIP) as logs:
        for core_id, record in sorted(report["core_records"].items()):
            if not record["photo_members"] or not record["log_members"]:
                continue
            display = []
            for kind, archive, members in (("Фото", photos, record["photo_members"]),
                                           ("Журнал", logs, record["log_members"])):
                thumbnails = []
                for member in members:
                    relative = preview(archive, member, "photo" if kind == "Фото" else "log")
                    thumbnails.append(f"<figure><a href='{html.escape(relative)}'><img loading='lazy' src='{html.escape(relative)}' alt='{kind} {html.escape(member)}'></a><figcaption>{html.escape(member)}</figcaption></figure>")
                display.append(f"<section><h3>{kind}</h3><div class='strip'>{''.join(thumbnails)}</div></section>")
            core = record["metadata"]
            cards.append(f"<details><summary><b>{html.escape(core_id)}</b> — {html.escape(core['site'])}, {len(record['photo_members'])} фото, {len(record['log_members'])} стр. журнала</summary>{''.join(display)}</details>")
    document = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS Florida Keys — фото и журналы</title><style>body{font:16px sans-serif;max-width:1200px;margin:2rem auto;color:#20313d;background:#f8fafb}details{background:white;border:1px solid #ccd;border-radius:.5rem;margin:1rem 0;padding:1rem}summary{cursor:pointer}.strip{display:flex;gap:1rem;overflow:auto}figure{margin:0;min-width:220px;max-width:360px}img{max-width:350px;max-height:400px}figcaption{font-size:.8rem;overflow-wrap:anywhere}h3{margin:.5rem 0}</style><h1>USGS Florida Keys: проверка фото и описаний</h1><p><a href='https://www.usgs.gov/data/descriptive-core-logs-core-photographs-radiocarbon-ages-and-accretion-data-holocene-reef-cores'>Официальный источник</a>, DOI 10.5066/F7NV9HJX, CC0. Совпадение ID керна подтверждено; границы глубин отдельных лотков и русские описания ещё не проверены. Ни одна строка не включена в обучение.</p>""" + "".join(cards) + "</html>"
    (ROOT / "review_index.html").write_text(document, encoding="utf-8")
    print(f"Review cards: {len(cards)}; previews: {len(list(PREVIEWS.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
