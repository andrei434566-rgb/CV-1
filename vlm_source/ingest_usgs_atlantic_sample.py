"""Inventory a small USGS Atlantic Coastal Plain core without inventing labels.

The source log is handwritten. Box-level depth spans are useful for review,
but are not sufficient to assign an interval description to any photo crop.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "external_raw" / "usgs_atlantic_coastal_plain"
ARCHIVE = DATA / "Witherspoon Island (DAR-237) Core folder.zip"
DICTIONARY = DATA / "Witherspoon Island (DAR-237) Data Dictionary.docx"
SOURCE_URL = "https://www.sciencebase.gov/catalog/item/668fe5ded34e511fa8d8ce2a"
PHOTO_RE = re.compile(r"Box (?P<box>\d+), (?P<top>\d+(?:\.\d+)?)-(?P<bottom>\d+(?:\.\d+)?) ft", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not ARCHIVE.is_file() or not DICTIONARY.is_file():
        raise FileNotFoundError("USGS archive and data dictionary must both be present")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC error: {bad}")
        rows = []
        for member in archive.infolist():
            match = PHOTO_RE.search(member.filename)
            if member.filename.startswith("__MACOSX/") or not match or not member.filename.lower().endswith(".jpg"):
                continue
            top = float(match["top"])
            bottom = float(match["bottom"])
            if bottom <= top:
                raise ValueError(f"Invalid box depths: {member.filename}")
            payload = archive.read(member)
            path = DATA / "photos" / f"box_{int(match['box']):02d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            with Image.open(path) as photo:
                photo.verify()
            rows.append({
                "box": int(match["box"]),
                "depth_top_ft_filename": top,
                "depth_bottom_ft_filename": bottom,
                "source_archive_member": member.filename,
                "image": str(path),
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "source_label_status": "none_at_photo_or_interval_level",
                "alignment_status": "box_depth_range_only; requires manual log/photo matching",
                "training_approved": False,
            })
    rows.sort(key=lambda item: item["box"])
    if len(rows) != 6 or [row["box"] for row in rows] != list(range(1, 7)):
        raise ValueError("Expected exactly six sequential box photographs")
    report = {
        "source": SOURCE_URL,
        "doi": "10.5066/P13TJBWJ",
        "license": "CC0 1.0 Universal on USGS release page",
        "well": "Witherspoon Island (DAR-237)",
        "archive_sha256": sha256(ARCHIVE),
        "dictionary_sha256": sha256(DICTIONARY),
        "log_pdf": "Witherspoon_stratigraphic_log.pdf",
        "log_pages": 6,
        "log_text_layer": False,
        "photos": rows,
        "approved_image_text_pairs": 0,
        "use_decision": "review_only; no VLM training until exact photo/log depth matching and transcription are checked",
    }
    (DATA / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (DATA / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "box", "depth_top_ft_filename", "depth_bottom_ft_filename",
            "image", "visible_depth_top_ft", "visible_depth_bottom_ft",
            "log_page", "source_description_verbatim", "reviewer", "review_status", "notes",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "pending" if key == "review_status" else "") for key in writer.fieldnames})
    cards = []
    for row in rows:
        image = Path(row["image"]).relative_to(DATA).as_posix()
        cards.append(
            f'<article><img src="{html.escape(image)}" loading="lazy">'
            f'<p>Ящик {row["box"]}; диапазон имени файла '
            f'{row["depth_top_ft_filename"]}–{row["depth_bottom_ft_filename"]} футов</p></article>'
        )
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>USGS Witherspoon — сверка фото и журнала</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#f3f5f7;color:#20252a}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}'
        'article{background:white;padding:12px;border-radius:8px}'
        'img{max-width:100%;max-height:430px;object-fit:contain}</style>'
        '<h1>Witherspoon Island (DAR-237)</h1>'
        '<p>USGS, CC0. Журнал: <a href="Witherspoon_stratigraphic_log.pdf">6 сканированных страниц</a>. '
        'Подпись диапазона в имени файла относится к ящику, не к каждой показанной колонке. '
        'Ни одного учебного соответствия фото–описание пока не утверждено. '
        'Для сверки используйте review.csv.</p><main>' + "".join(cards) + '</main></html>'
    )
    (DATA / "review.html").write_text(page, encoding="utf-8")
    ingest_cheraw()
    print(json.dumps({"photos": len(rows), "approved_pairs": 0, "archive_sha256": report["archive_sha256"]}))


def ingest_cheraw() -> None:
    """Stage the second USGS well at box level, with no inferred lithology."""
    archive_path = DATA / "Cheraw State Park (CTF-81) Core folder.zip"
    dictionary_path = DATA / "Cheraw State Park (CTF-81) Data Dictionary.docx"
    if not archive_path.is_file() or not dictionary_path.is_file():
        return
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Cheraw archive CRC error: {bad}")
        log_name = "Cheraw State Park (CTF-81) Stratigraphic Log.PDF"
        (DATA / "Cheraw_stratigraphic_log.pdf").write_bytes(archive.read(log_name))
        for member in archive.infolist():
            match = PHOTO_RE.search(member.filename)
            if member.filename.startswith("__MACOSX/") or not match or not member.filename.lower().endswith(".jpg"):
                continue
            payload = archive.read(member)
            path = DATA / "photos" / "cheraw" / f"box_{int(match['box']):02d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            with Image.open(path) as photo:
                photo.verify()
            rows.append({
                "box": int(match["box"]),
                "depth_top_ft_filename": float(match["top"]),
                "depth_bottom_ft_filename": float(match["bottom"]),
                "source_archive_member": member.filename,
                "image": str(path),
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "alignment_status": "box_depth_range_only; requires manual log/photo matching",
                "training_approved": False,
            })
    rows.sort(key=lambda row: row["box"])
    if [row["box"] for row in rows] != list(range(1, 26)):
        raise ValueError("Expected 25 sequential Cheraw box photos")
    report = {
        "source": SOURCE_URL,
        "doi": "10.5066/P13TJBWJ",
        "license": "CC0 1.0 Universal on USGS release page",
        "well": "Cheraw State Park (CTF-81)",
        "archive_sha256": sha256(archive_path),
        "dictionary_sha256": sha256(dictionary_path),
        "log_pdf": "Cheraw_stratigraphic_log.pdf",
        "log_pages": 3,
        "log_text_layer": False,
        "photos": rows,
        "approved_image_text_pairs": 0,
        "use_decision": "review_only; no VLM training until exact photo/log depth matching and transcription are checked",
    }
    (DATA / "cheraw_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["box", "depth_top_ft_filename", "depth_bottom_ft_filename", "image", "visible_depth_top_ft",
              "visible_depth_bottom_ft", "log_page", "source_description_verbatim", "reviewer", "review_status", "notes"]
    with (DATA / "cheraw_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "pending" if key == "review_status" else "") for key in fields})
    cards = []
    for row in rows:
        image = Path(row["image"]).relative_to(DATA).as_posix()
        cards.append(f'<article><img src="{html.escape(image)}" loading="lazy"><p>Ящик {row["box"]}; '
                     f'{row["depth_top_ft_filename"]}–{row["depth_bottom_ft_filename"]} футов</p></article>')
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>USGS Cheraw — сверка</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#f3f5f7;color:#20252a}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}'
        'article{background:white;padding:12px;border-radius:8px}img{max-width:100%;max-height:430px;object-fit:contain}</style>'
        '<h1>Cheraw State Park (CTF-81)</h1><p>USGS, CC0. '
        '<a href="Cheraw_stratigraphic_log.pdf">Три страницы журнала</a>. '
        'Диапазон глубин дан для ящика; точных учебных пар пока нет. Правки: cheraw_review.csv.</p><main>'
        + "".join(cards) + '</main></html>'
    )
    (DATA / "cheraw_review.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
