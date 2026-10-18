"""Inventory source-linked Biscayne aquifer core photos and lithologic logs.

This creates review records only. A shared borehole ID does not establish which
pixels in a multi-column core box correspond to each described depth interval.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "external_raw" / "usgs_biscayne_2014"
REVIEW = SOURCE / "review"
LOG = SOURCE / "sir2014-5138_appendix01.pdf"
PAGE = "https://pubs.usgs.gov/sir/2014/5138/"
BOREHOLE = re.compile(r"G[^0-9]?(38\d{2})")
PHOTO_NAME = re.compile(r"sir2014-\d+_appendix02_g(38\d{2})\.pdf$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not LOG.is_file():
        raise FileNotFoundError(LOG)
    photos = sorted(path for path in SOURCE.glob("*.pdf") if PHOTO_NAME.match(path.name))
    if len(photos) != 9:
        raise ValueError(f"Expected nine source photo PDFs, found {len(photos)}")

    source_pages: dict[str, list[dict]] = {}
    log_reader = PdfReader(LOG)
    for page_number, page in enumerate(log_reader.pages, 1):
        page_text = page.extract_text() or ""
        match = BOREHOLE.search(page_text[:250])
        if match is None:
            raise ValueError(f"No borehole ID in source log page {page_number}")
        hole = "G-" + match.group(1)
        source_pages.setdefault(hole, []).append({
            "page_number": page_number,
            "source_text_en": page_text,
        })

    REVIEW.mkdir(exist_ok=True)
    records = []
    for photo in photos:
        match = PHOTO_NAME.match(photo.name)
        assert match is not None
        hole = "G-" + match.group(1)
        if hole not in source_pages:
            raise ValueError(f"No lithologic description for {hole}")
        reader = PdfReader(photo)
        photo_pages = []
        for page_number, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            photo_hole = BOREHOLE.search(page_text[:250])
            if photo_hole is None or photo_hole.group(1) != match.group(1):
                raise ValueError(f"Photo page {page_number} does not show {hole}")
            photo_pages.append({"page_number": page_number, "embedded_image_count": len(page.images)})
        if not reader.pages[0].images:
            raise ValueError(f"No embedded core photograph for {hole}")
        first_image = reader.pages[0].images[0]
        with Image.open(io.BytesIO(first_image.data)) as image:
            preview = image.convert("RGB")
            preview.thumbnail((450, 750))
            preview.save(REVIEW / f"{hole}.jpg", quality=82)
        text = "\n".join(page["source_text_en"] for page in source_pages[hole])
        records.append({
            "source_id": "usgs_sir2014_5138_biscayne",
            "source_url": PAGE,
            "borehole_id": hole,
            "lithologic_log_pdf": str(LOG.relative_to(ROOT)),
            "lithologic_log_pages": [page["page_number"] for page in source_pages[hole]],
            "source_description_en": text,
            "photo_pdf": str(photo.relative_to(ROOT)),
            "photo_pdf_pages": photo_pages,
            "preview": str((REVIEW / f"{hole}.jpg").relative_to(ROOT)),
            "pairing_basis": "Same explicit borehole ID in official Appendix 1 and official Appendix 2.",
            "pixel_to_depth_alignment": "not_verified_multi_column_core_boxes",
            "russian_translation_status": "pending",
            "training_approved": False,
        })

    if set(source_pages) != {record["borehole_id"] for record in records}:
        raise ValueError("Photo and description borehole sets differ")
    target = SOURCE / "source_direct_boreholes_quarantine.jsonl"
    target.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    summary = {
        "source_id": "usgs_sir2014_5138_biscayne",
        "source_url": PAGE,
        "appendix_1_url": PAGE + "appendix/sir2014-5138_appendix01.pdf",
        "appendix_2_index_url": PAGE + "appendix/sir2014-5138_appendix02/",
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in [LOG, *photos]],
        "description_pdf_pages": len(log_reader.pages),
        "matched_boreholes": len(records),
        "core_photo_pdf_pages": sum(len(record["photo_pdf_pages"]) for record in records),
        "training_approved_pairs": 0,
        "reason_not_trained": "Borehole match is exact, but the multi-column photographs have handwritten and discontinuous depth markers; interval-to-pixel correspondence and Russian translation are unverified.",
        "records": str(target.relative_to(ROOT)),
    }
    (SOURCE / "inventory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for record in records:
        hole = record["borehole_id"]
        excerpt = re.sub(r"\s+", " ", record["source_description_en"])
        cards.append(
            f'<article><h2>{hole}</h2><img src="{hole}.jpg" alt="Первое фото керна {hole}">'
            f'<p>{len(record["photo_pdf_pages"])} страниц фото; {len(record["lithologic_log_pages"])} страниц описания.</p>'
            f'<p><a href="../{html.escape(Path(record["photo_pdf"]).name)}">Исходные фото PDF</a> · '
            f'<a href="../{LOG.name}">Исходное описание PDF</a> (стр. {record["lithologic_log_pages"][0]}–{record["lithologic_log_pages"][-1]})</p>'
            f'<p>{html.escape(excerpt[:700])}…</p></article>'
        )
    (REVIEW / "index.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>USGS Biscayne: фото и описания</title>'
        '<style>body{font:16px system-ui;margin:2rem;max-width:1200px}article{border:1px solid #bbb;padding:1rem;margin:1rem 0;overflow:auto}img{float:left;max-width:260px;max-height:420px;margin-right:1.5rem}</style>'
        '<h1>USGS Biscayne: пары по скважине для проверки</h1><p>Глубина внутри фотографии не выверена. В обучение записи не допущены.</p>'
        + "".join(cards) + "</html>", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("matched_boreholes", "description_pdf_pages", "core_photo_pdf_pages", "training_approved_pairs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
