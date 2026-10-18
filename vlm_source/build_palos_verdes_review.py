"""Build source-linked review for Palos Verdes photo sections and graphic logs."""

from __future__ import annotations

import html
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).parent / "data" / "external_raw" / "usgs_palos_verdes"
PHOTOS = ROOT / "PalosVerdesCores_Photos.zip"
FIGURES = ROOT / "PalosVerdesCores_Figures.pdf"
PREVIEWS = ROOT / "review_previews"


def normalize_core_id(name: str) -> str | None:
    match = re.search(r"DR\s*(\d+)\s*VC-(\d+)(?!\d)|PV(\d+)-G(\d+)(?!\d)", name.upper())
    if not match:
        return None
    if match.group(1):
        return f"DR{match.group(1)}VC-{match.group(2)}"
    return f"PV{match.group(3)}-G{match.group(4)}"


def main() -> None:
    PREVIEWS.mkdir(exist_ok=True)
    reader = PdfReader(FIGURES)
    page_by_core: dict[str, int] = {}
    for page, item in enumerate(reader.pages, start=1):
        core = normalize_core_id(item.extract_text() or "")
        if not core:
            raise RuntimeError(f"Core ID missing on graphic log page {page}")
        if core in page_by_core:
            raise RuntimeError(f"Duplicate core ID {core}")
        page_by_core[core] = page
    photos = []
    with zipfile.ZipFile(PHOTOS) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".bmp") or member.startswith("__MACOSX/"):
                continue
            core = normalize_core_id(member)
            if not core:
                raise RuntimeError(f"Core ID missing in photo {member}")
            preview = PREVIEWS / (Path(member).stem + ".jpg")
            with archive.open(member) as source:
                image = Image.open(io.BytesIO(source.read())).convert("RGB")
                dimensions = list(image.size)
                if not preview.exists():
                    image.thumbnail((2200, 2200))
                    image.save(preview, quality=88, optimize=True)
            depth = re.search(r"(?<!\d)(\d+)-(\d+)cm", member, flags=re.I)
            photos.append({
                "core_id": core,
                "source_zip_member": member,
                "source_pixel_dimensions": dimensions,
                "section_depth_cm_from_filename": [int(depth.group(1)), int(depth.group(2))] if depth else None,
                "graphic_log_pdf_page_1_based": page_by_core.get(core),
                "preview": preview.relative_to(ROOT).as_posix(),
                "training_approved": False,
            })
    photos.sort(key=lambda item: (item["core_id"], item["source_zip_member"]))
    audit = {
        "source_url": "https://www.usgs.gov/data/sediment-core-data-offshore-palos-verdes-california",
        "pdf_pages": len(reader.pages),
        "distinct_log_core_ids": len(page_by_core),
        "source_photo_sections": len(photos),
        "photo_sections_with_matching_log_page": sum(item["graphic_log_pdf_page_1_based"] is not None for item in photos),
        "photo_sections_with_depth_range_in_filename": sum(item["section_depth_cm_from_filename"] is not None for item in photos),
        "training_approved": False,
        "review_needed": "Confirm depth alignment and lithologic boundaries on each PDF page; translate source labels into Russian and check visible features.",
        "photos": photos,
    }
    (ROOT / "alignment_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for photo in photos:
        page = photo["graphic_log_pdf_page_1_based"]
        link = f"PalosVerdesCores_Figures.pdf#page={page}" if page else "#"
        depth = photo["section_depth_cm_from_filename"]
        rows.append(
            f"<tr><td>{html.escape(photo['core_id'])}</td><td>{html.escape(Path(photo['source_zip_member']).name)}</td>"
            f"<td>{depth or 'не указана'}</td><td><a href='{html.escape(link)}'>Журнал, стр. {page or '?'}</a></td>"
            f"<td><a href='{html.escape(photo['preview'])}'><img loading='lazy' src='{html.escape(photo['preview'])}' width='85'></a></td></tr>"
        )
    document = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS Palos Verdes — проверка пар</title>
<style>body{font:16px sans-serif;max-width:1150px;margin:2rem auto;color:#25313b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:.5rem;text-align:left;vertical-align:top}th{background:#e9eff2}tr:nth-child(even){background:#f7f9fa}img{max-height:150px;object-fit:contain}</style>
<h1>USGS Palos Verdes: фотографии и графические журналы</h1>
<p>Источник: <a href='https://www.usgs.gov/data/sediment-core-data-offshore-palos-verdes-california'>USGS, DOI 10.5066/F7G44NB6</a>. Лицензия релиза CC0. Все строки требуют ручной проверки глубины и литологических границ; в обучение не включены.</p>
<table><tr><th>Керн</th><th>Исходное фото в ZIP</th><th>Глубина, см, из имени файла</th><th>Журнал</th><th>Предпросмотр</th></tr>""" + "".join(rows) + "</table></html>"
    (ROOT / "review_index.html").write_text(document, encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("pdf_pages", "distinct_log_core_ids", "source_photo_sections", "photo_sections_with_matching_log_page", "photo_sections_with_depth_range_in_filename")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
