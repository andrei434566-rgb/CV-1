"""Link Monterey Canyon photo sections to their official graphic-log pages."""

from __future__ import annotations

import html
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).parent / "data" / "external_raw" / "usgs_monterey_canyon"
ZIP = ROOT / "NorthernFlankMontereyCanyonCores_Photos.zip"
PDF = ROOT / "NorthernFlankMontereyCanyonCores_Figures.pdf"
PREVIEWS = ROOT / "review_previews"
SOURCE = "https://www.usgs.gov/data/sediment-core-data-northern-flank-monterey-canyon-offshore-california"


def core_id(text: str) -> str | None:
    match = re.search(r"DR\s*(\d+)[\s_]*VC-(\d+)(?!\d)", text.upper())
    return f"DR{match.group(1)}VC-{match.group(2)}" if match else None


def main() -> None:
    PREVIEWS.mkdir(exist_ok=True)
    pages = PdfReader(PDF).pages
    page_map = {}
    for number, page in enumerate(pages, 1):
        key = core_id(page.extract_text() or "")
        if not key or key in page_map:
            raise RuntimeError(f"Missing or repeated core ID on log page {number}")
        page_map[key] = number
    photos = []
    with zipfile.ZipFile(ZIP) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".bmp") or member.startswith("__MACOSX/"):
                continue
            key = core_id(member)
            if key not in page_map:
                raise RuntimeError(f"Photo has no exact core ID match in logs: {member}")
            depth = re.search(r"(\d+)-(\d+)cm", member, flags=re.I)
            if not depth:
                raise RuntimeError(f"Photo has no section depth range: {member}")
            preview = PREVIEWS / (Path(member).stem + ".jpg")
            with archive.open(member) as stream:
                image = Image.open(io.BytesIO(stream.read())).convert("RGB")
                dimensions = list(image.size)
                if not preview.exists():
                    image.thumbnail((2200, 2200))
                    image.save(preview, quality=88, optimize=True)
            photos.append({
                "core_id": key,
                "source_zip_member": member,
                "source_pixel_dimensions": dimensions,
                "section_depth_cm_from_filename": [int(depth.group(1)), int(depth.group(2))],
                "graphic_log_pdf_page_1_based": page_map[key],
                "preview": preview.relative_to(ROOT).as_posix(),
                "training_approved": False,
            })
    photos.sort(key=lambda item: (item["core_id"], item["section_depth_cm_from_filename"][0]))
    audit = {
        "source_url": SOURCE,
        "pdf_pages": len(pages),
        "distinct_log_core_ids": len(page_map),
        "source_photo_sections": len(photos),
        "photos_with_exact_core_id_and_depth_range": len(photos),
        "training_approved": False,
        "review_needed": "Manually verify graphic lithologic boundaries for each photo section and translate source terms into Russian.",
        "photos": photos,
    }
    (ROOT / "alignment_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for item in photos:
        member = html.escape(item["source_zip_member"])
        preview = html.escape(item["preview"])
        page = item["graphic_log_pdf_page_1_based"]
        depth = item["section_depth_cm_from_filename"]
        rows.append(f"<tr><td>{item['core_id']}</td><td>{member}</td><td>{depth[0]}–{depth[1]}</td><td><a href='{PDF.name}#page={page}'>Журнал, стр. {page}</a></td><td><a href='{preview}'><img loading='lazy' src='{preview}' width='85'></a></td></tr>")
    output = f"""<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS Monterey Canyon — проверка пар</title><style>body{{font:16px sans-serif;max-width:1150px;margin:2rem auto;color:#25313b}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd;padding:.5rem;text-align:left;vertical-align:top}}th{{background:#e9eff2}}tr:nth-child(even){{background:#f7f9fa}}img{{max-height:150px;object-fit:contain}}</style><h1>USGS Monterey Canyon: фото и журналы</h1><p><a href='{SOURCE}'>Официальный релиз USGS, DOI 10.5066/F7J67F2K</a>, CC0. Привязка по ID керна и имени фото — лишь начало проверки: метки пород и границы должны быть подтверждены по графическому журналу.</p><table><tr><th>Керн</th><th>Исходный файл в ZIP</th><th>Глубина, см</th><th>Журнал</th><th>Предпросмотр</th></tr>{''.join(rows)}</table></html>"""
    (ROOT / "review_index.html").write_text(output, encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("pdf_pages", "distinct_log_core_ids", "source_photo_sections", "photos_with_exact_core_id_and_depth_range")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
