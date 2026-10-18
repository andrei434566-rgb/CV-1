"""Extract user-provided geology references into a page-indexed local corpus.

The raw PDF remains at its original path. This index is intended for local
search/retrieval and provenance; it is not by itself a supervised target set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from datetime import date
from pathlib import Path

from pypdf import PdfReader


PROJECT = Path(__file__).resolve().parent
DEFAULT_PDF = Path(r"E:\Газпром Фациал модели.pdf")
DEFAULT_OUTPUT = PROJECT / "data" / "knowledge" / "gazprom_facies_models_pages.jsonl"
SOURCE_MANIFEST = PROJECT / "data" / "knowledge" / "gazprom_facies_models_source.json"

# Printed page bounds from the book's table of contents, inclusive.
CHAPTERS = [
    (1, 1, 34, "Фации, фациальные модели и современные принципы стратиграфии"),
    (2, 35, 58, "Влияние колебаний уровня моря"),
    (3, 59, 101, "Фациальный анализ по данным глубинных исследований"),
    (4, 102, 164, "Фациальные модели по данным палеонтологического анализа"),
    (5, 165, 219, "Ледниковые образования"),
    (6, 220, 258, "Вулканогенно-обломочные породы"),
    (7, 259, 314, "Аллювиальные отложения"),
    (8, 315, 345, "Эоловые системы"),
    (9, 346, 393, "Дельты"),
    (10, 394, 429, "Трансгрессивные системы барьерных островов и эстуариев"),
    (11, 430, 482, "Осадочные системы приливно-отливных зон"),
    (12, 483, 528, "Мелководные морские системы с преобладанием волн и штормов"),
    (13, 529, 587, "Турбидиты и подводные конусы выноса"),
    (14, 588, 613, "Введение в модели карбонатных и эвапоритовых фаций"),
    (15, 614, 669, "Мелководноморские карбонатные платформы"),
    (16, 670, 712, "Карбонатные отложения приливно-отливной зоны"),
    (17, 713, 767, "Рифы и органогенные постройки"),
    (18, 768, 820, "Карбонатные склоны"),
    (19, 821, 910, "Эвапориты"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw or "")
    text = text.replace("\u00ad", "").replace("\ufffd", " ")
    text = re.sub(r"([A-Za-zА-Яа-яЁё])[-\u2010\u2011]\s*\n\s*([а-яё])", r"\1\2", text)
    lines = []
    for line in text.replace("\r", "\n").split("\n"):
        line = re.sub(r"[\t ]+", " ", line).strip()
        # PDF font artifacts from vector figures are not useful text.
        line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]+", " ", line)
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    text = "\n".join(lines).strip()
    text = re.sub(r"(?<=[а-яё])\n(?=[а-яё])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def chapter_for(book_page: int | None) -> tuple[int | None, str | None]:
    if book_page is None:
        return None, None
    for number, start, end, title in CHAPTERS:
        if start <= book_page <= end:
            return number, title
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"Source PDF not found: {args.pdf}")

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extracted_pages = 0
    text_characters = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for index, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                raw = page.extract_text() or ""
            text = clean_text(raw)
            # The cover starts at PDF p. 16, where the printed book page is 1.
            book_page = index - 15 if index >= 16 else None
            chapter_number, chapter_title = chapter_for(book_page)
            status = "text" if len(text) >= 120 else "sparse_or_image_only"
            extracted_pages += status == "text"
            text_characters += len(text)
            row = {
                "source_id": "user_gazprom_facies_models_2017_ru",
                "pdf_page": index,
                "book_page": book_page,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "extraction_status": status,
                "text_characters": len(text),
                "text": text,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                print(f"Indexed {index}/{len(reader.pages)} PDF pages", flush=True)

    manifest = {
        "source_id": "user_gazprom_facies_models_2017_ru",
        "title": "Фациальные модели",
        "editors": ["Роджер Уолкер", "Ноэль Джеймс"],
        "publication": "Ижевск: Институт компьютерных исследований, 2017; русское издание перевода 2016 года",
        "original_copyright": "1992 Geological Association of Canada; Russian translation copyright 2016, Ижевский институт компьютерных исследований",
        "provided_path": str(args.pdf),
        "sha256": sha256(args.pdf),
        "pdf_pages": len(reader.pages),
        "indexed_pages": len(reader.pages),
        "pages_with_extractable_text_ge_120_chars": extracted_pages,
        "extracted_text_characters": text_characters,
        "indexed_jsonl": str(args.output.relative_to(PROJECT)),
        "extraction_date": date.today().isoformat(),
        "use_note": "User-provided copyrighted reference. Retain locally for this user's project; do not redistribute the extracted text. Image-only pages and figures are not OCR-labeled by this text extractor.",
    }
    SOURCE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pdf_pages": len(reader.pages),
        "pages_with_text": extracted_pages,
        "extracted_characters": text_characters,
        "indexed_jsonl": str(args.output),
        "source_manifest": str(SOURCE_MANIFEST),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
