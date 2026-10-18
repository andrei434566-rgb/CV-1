"""Stage three paired image-log/photo groups from USGS OFR 2012-1095."""

from __future__ import annotations

import hashlib
import html
import io
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "external_raw" / "usgs_dry_tortugas_2012"
REPORT = SOURCE / "ofr2012-1095.pdf"
REVIEW = SOURCE / "review"
OFFICIAL = "https://pubs.usgs.gov/of/2012/1095/"
GROUPS = (
    ("East Key", 25, (26,)),
    ("Northkey Harbor Shallow", 27, (28, 29)),
    ("Northkey Harbor Deep", 30, (31,)),
)


def main() -> None:
    reader = PdfReader(REPORT)
    REVIEW.mkdir(exist_ok=True)
    records = []
    cards = []
    for site, log_page_number, photo_pages in GROUPS:
        log_page = reader.pages[log_page_number - 1]
        if site not in (log_page.extract_text() or ""):
            raise ValueError(f"Missing source site caption in log page {log_page_number}: {site}")
        if not log_page.images:
            raise ValueError(f"No image log for {site}")
        slug = site.lower().replace(" ", "_")
        first_photo = None
        for page_number in photo_pages:
            page = reader.pages[page_number - 1]
            if site not in (page.extract_text() or "") or not page.images:
                raise ValueError(f"No matched source photo on page {page_number}: {site}")
            with Image.open(io.BytesIO(page.images[0].data)) as image:
                output = REVIEW / f"{slug}_photo_{page_number}.jpg"
                image.convert("RGB").save(output, quality=93)
                if first_photo is None:
                    first_photo = output
        records.append({
            "source_id": "usgs_ofr2012_1095_dry_tortugas",
            "source_url": OFFICIAL,
            "site": site,
            "report_pdf": str(REPORT.relative_to(ROOT)),
            "log_pdf_page": log_page_number,
            "photo_pdf_pages": list(photo_pages),
            "first_photo": str(first_photo.relative_to(ROOT)),
            "pairing_basis": "Exact site name in adjacent official report log and photo page captions.",
            "log_text_status": "image_only_not_transcribed",
            "pixel_to_depth_alignment": "not_verified_multi_column_box",
            "training_approved": False,
        })
        cards.append(f'<article><h2>{html.escape(site)}</h2><img src="{html.escape(first_photo.name)}">'
                     f'<p>Описание: стр. PDF {log_page_number}; фото: стр. PDF {", ".join(map(str, photo_pages))}.</p></article>')
    manifest = SOURCE / "source_direct_sites_quarantine.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "source_id": "usgs_ofr2012_1095_dry_tortugas",
        "source_url": OFFICIAL,
        "report_pdf": str(REPORT.relative_to(ROOT)),
        "report_bytes": REPORT.stat().st_size,
        "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        "report_pages": len(reader.pages),
        "matched_sites": len(records),
        "photo_pages": sum(len(row["photo_pdf_pages"]) for row in records),
        "training_approved_pairs": 0,
        "reason_not_trained": "Lithologic log is an embedded image; no verified transcription, Russian translation or interval-to-pixel mapping.",
        "records": str(manifest.relative_to(ROOT)),
    }
    (SOURCE / "inventory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (REVIEW / "index.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>USGS Dry Tortugas</title>'
        '<style>body{font:16px system-ui;margin:2rem;max-width:1100px}article{border:1px solid #aaa;padding:1rem;margin:1rem 0}img{max-width:300px;max-height:500px}</style>'
        '<h1>Dry Tortugas: исходные фото и журналы</h1><p>Прямое соответствие по участку; обучение не разрешено.</p>'
        + "".join(cards) + "</html>", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("matched_sites", "photo_pages", "training_approved_pairs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
