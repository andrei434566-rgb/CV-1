"""Bounded, read-only keyword search in folders chosen by the geologist."""

from __future__ import annotations

import os
import re
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree


TEXT_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".log"}
DOCUMENT_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx", ".xlsx", ".xlsm", ".xls"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ARCHIVE_SUFFIXES = {".zip"}
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", "model_cache", "node_modules", "outputs"}
MAX_FILES = 2000
MAX_ENTRIES = 50_000
MAX_BYTES = 25 * 1024 * 1024
MAX_CHARS = 120_000

SEARCH_STOPWORDS = {
    "найди", "найдите", "найти", "поищи", "поищите", "ищи", "ищите", "поиск",
    "информация", "информацию", "информации", "данные", "материалы", "материал",
    "файл", "файлы", "файлах", "папка", "папке", "папках", "скважина", "скважине",
    "скважины", "скважину", "скважин", "пожалуйста", "какая", "какой", "какие",
    "по", "про", "для", "где", "что", "есть", "это", "мне", "все", "всё",
}

PHOTO_QUERY_STOPWORDS = SEARCH_STOPWORDS | {
    "дай", "дайте", "пришли", "пришлите", "скинь", "ссылку", "ссылка",
    "ссылки", "путь", "адрес", "фото", "фотография", "фотографии",
    "снимок", "снимки", "изображение", "изображения", "открой", "откройте",
}
WELL_CODE_RE = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9])([A-Za-zА-Яа-яЁё]{1,8})\s*[-_.–—№#]?\s*(\d{1,5})(?!\d)"
)


def _well_codes(value: str) -> set[tuple[str, int]]:
    codes = set()
    for match in WELL_CODE_RE.finditer(value):
        prefix = match.group(1).casefold().replace("ё", "е")
        # Cyrillic Р and Latin R can represent the same well prefix. Latin P
        # remains distinct: the public P25 cores are unrelated to well Р-25.
        if len(prefix) == 1 and prefix in {"р", "r"}:
            prefix = "r"
        codes.add((prefix, int(match.group(2))))
    return codes


def search_photo_files(
    query: str,
    folders: list[str],
    known_photos: list[tuple[str, str]] | None = None,
    limit: int = 8,
    stats: dict[str, int | bool] | None = None,
) -> list[dict[str, str]]:
    """Find real image files by well code in names, folders or project records."""
    requested_codes = _well_codes(query)
    terms = [
        word for word in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{3,}", query.casefold())
        if word not in PHOTO_QUERY_STOPWORDS and not word.isdigit()
    ]
    if not requested_codes and not terms:
        if stats is not None:
            stats.update(documents_read=0, entries_seen=0, truncated=False)
        return []

    candidates: dict[str, tuple[int, dict[str, str]]] = {}
    seen_files = 0
    truncated = False

    def consider(path: Path, well_name: str = "") -> None:
        if path.suffix.casefold() not in IMAGE_SUFFIXES or not path.is_file():
            return
        path_codes = _well_codes(str(path))
        record_codes = _well_codes(well_name)
        if requested_codes:
            if requested_codes & record_codes:
                score = 200
            elif requested_codes & _well_codes(path.stem):
                score = 160
            elif requested_codes & path_codes:
                score = 120
            else:
                return
        else:
            searchable = (str(path) + " " + well_name).casefold()
            matches = sum(term in searchable for term in terms)
            if not matches:
                return
            score = matches * 10
        key = str(path).casefold()
        hit = {
            "path": str(path),
            "excerpt": f"Фото скважины {well_name}" if well_name else "Фото найдено по имени файла или папки",
            "kind": "image",
        }
        if key not in candidates or score > candidates[key][0]:
            candidates[key] = (score, hit)

    for path_text, well_name in known_photos or []:
        consider(Path(path_text), well_name)
    for folder in folders[:12]:
        root = Path(folder).expanduser()
        if not root.is_dir():
            continue
        for current, dirs, names in os.walk(root):
            dirs[:] = [name for name in dirs if name.casefold() not in SKIP_PARTS]
            for name in names:
                seen_files += 1
                if seen_files > MAX_ENTRIES:
                    truncated = True
                    break
                if Path(name).suffix.casefold() in IMAGE_SUFFIXES:
                    consider(Path(current) / name)
            if truncated:
                break
        if truncated:
            break
    if stats is not None:
        stats.update(documents_read=0, entries_seen=seen_files, truncated=truncated)
    ranked = sorted(candidates.values(), key=lambda item: (-item[0], item[1]["path"].casefold()))
    return [hit for _, hit in ranked[:max(1, min(limit, 20))]]


def _document_text(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8-sig", errors="replace")[:MAX_CHARS]
    if suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            if archive.getinfo("word/document.xml").file_size > 12 * 1024 * 1024:
                return ""
            xml = ElementTree.fromstring(archive.read("word/document.xml"))
        paragraphs = []
        for paragraph in xml.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
            paragraphs.append("".join(node.text or "" for node in paragraph.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
            )))
        return "\n".join(paragraphs)[:MAX_CHARS]
    if suffix in {".xlsx", ".xlsm"}:
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        with zipfile.ZipFile(path) as archive:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                if archive.getinfo("xl/sharedStrings.xml").file_size > 12 * 1024 * 1024:
                    return ""
                strings = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = ["".join(text.text or "" for text in item.iter(ns + "t"))
                          for item in strings.iter(ns + "si")]
            rows = []
            length = 0
            sheet_names = sorted(name for name in archive.namelist()
                                 if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))[:5]
            for name in sheet_names:
                if archive.getinfo(name).file_size > 12 * 1024 * 1024:
                    continue
                sheet = ElementTree.fromstring(archive.read(name))
                for row in sheet.iter(ns + "row"):
                    cells = []
                    for cell in row.iter(ns + "c"):
                        value = cell.find(ns + "v")
                        if cell.attrib.get("t") == "s" and value is not None:
                            index = int(value.text or "-1")
                            text = shared[index] if 0 <= index < len(shared) else ""
                        elif cell.attrib.get("t") == "inlineStr":
                            text = "".join(node.text or "" for node in cell.iter(ns + "t"))
                        else:
                            text = value.text if value is not None else ""
                        if text:
                            cells.append(text)
                    if cells:
                        line = " ".join(cells)
                        rows.append(line)
                        length += len(line)
                    if length >= MAX_CHARS:
                        return "\n".join(rows)[:MAX_CHARS]
            return "\n".join(rows)[:MAX_CHARS]
    if suffix == ".xls":
        try:
            import xlrd
        except ImportError:
            vendor = Path(__file__).resolve().parents[2] / "third_party"
            if not (vendor / "xlrd").is_dir():
                raise
            sys.path.insert(0, str(vendor))
            import xlrd
        workbook = xlrd.open_workbook(str(path), on_demand=True)
        try:
            rows = []
            length = 0
            for sheet in workbook.sheets()[:5]:
                for row_index in range(sheet.nrows):
                    line = " ".join(str(value).strip() for value in sheet.row_values(row_index) if str(value).strip())
                    if line:
                        rows.append(line)
                        length += len(line)
                    if length >= MAX_CHARS:
                        return "\n".join(rows)[:MAX_CHARS]
            return "\n".join(rows)[:MAX_CHARS]
        finally:
            workbook.release_resources()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            # The source launcher can use the small bundled offline wheel;
            # release builds install pypdf from requirements.txt.
            vendor = Path(__file__).resolve().parents[2] / "third_party"
            if not (vendor / "pypdf").is_dir():
                raise
            sys.path.insert(0, str(vendor))
            from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:80])[:MAX_CHARS]
    return ""


@lru_cache(maxsize=256)
def _cached_document_text(path: str, modified_ns: int, size: int) -> str:
    return _document_text(Path(path))


def search_folders(query: str, folders: list[str], limit: int = 5,
                   stats: dict[str, int | bool] | None = None) -> list[dict[str, str]]:
    """Return only short, attributed excerpts. Never execute or modify files."""
    terms = list(dict.fromkeys(
        word[:5] if len(word) > 6 else word
        for word in re.findall(r"[\wа-яё]{3,}", query.casefold())
        if word not in SEARCH_STOPWORDS
    ))[:12]
    if not terms:
        terms = re.findall(r"[\wа-яё]{3,}", query.casefold())[:4]
    if not terms:
        if stats is not None:
            stats.update(documents_read=0, entries_seen=0, truncated=False)
        return []
    candidates: list[tuple[int, dict[str, str]]] = []
    scanned = 0
    visited_total = 0
    truncated = False
    numeric_terms = [term for term in terms if term.isdigit()]
    required_matches = 2 if numeric_terms and len(terms) >= 2 else 1
    for folder in folders[:12]:
        root = Path(folder).expanduser()
        if not root.is_dir():
            continue
        visited = 0
        for current, dirs, names in os.walk(root):
            dirs[:] = [name for name in dirs if name.casefold() not in SKIP_PARTS]
            for name in names:
                visited += 1
                visited_total += 1
                if scanned >= MAX_FILES or visited >= MAX_ENTRIES:
                    truncated = True
                    break
                path = Path(current) / name
                if name.startswith(("theory_cards_", "theory_training_", "theory_holdout_")):
                    continue
                suffix = path.suffix.casefold()
                if suffix not in DOCUMENT_SUFFIXES | IMAGE_SUFFIXES | ARCHIVE_SUFFIXES:
                    continue
                relative_name = str(path.relative_to(root)).casefold()
                if suffix in ARCHIVE_SUFFIXES:
                    try:
                        with zipfile.ZipFile(path) as archive:
                            for member in archive.infolist()[:5000]:
                                visited += 1
                                visited_total += 1
                                if visited >= MAX_ENTRIES:
                                    truncated = True
                                    break
                                if member.is_dir():
                                    continue
                                member_name = member.filename.casefold()
                                searchable_name = relative_name + "/" + member_name
                                matched = [term for term in terms if term in searchable_name]
                                if len(matched) < required_matches or any(term not in searchable_name for term in numeric_terms):
                                    continue
                                score = 6 * len(matched) + 6 * len([term for term in matched if term.isdigit()])
                                candidates.append((score, {
                                    "path": str(path),
                                    "excerpt": f"Внутри архива: {member.filename}. Совпадение по названию; содержимое файла не прочитано.",
                                    "kind": "archive",
                                }))
                    except (OSError, ValueError, zipfile.BadZipFile):
                        pass
                    continue
                if suffix in IMAGE_SUFFIXES:
                    matched = [term for term in terms if term in relative_name]
                    if len(matched) >= required_matches and all(term in relative_name for term in numeric_terms):
                        score = 6 * len(matched) + 6 * len([term for term in matched if term.isdigit()])
                        candidates.append((score, {
                            "path": str(path), "excerpt": "Фото найдено по названию файла или папки; текстового описания в нём нет.",
                            "kind": "image",
                        }))
                    continue
                try:
                    if path.is_symlink():
                        continue
                    stat = path.stat()
                    if stat.st_size > MAX_BYTES:
                        continue
                    scanned += 1
                    try:
                        content = _cached_document_text(str(path), stat.st_mtime_ns, stat.st_size)
                    except Exception:
                        # A PDF parser may not be installed, or a document may
                        # be damaged. The filename can still be searched.
                        content = ""
                except (OSError, ValueError, ImportError, RuntimeError, KeyError,
                        zipfile.BadZipFile, ElementTree.ParseError):
                    continue
                title = path.name.casefold()
                is_theory = path.name.startswith("theory_") and path.name.endswith("_ru.md")
                sections = re.split(r"(?=^## )", content, flags=re.MULTILINE) if is_theory else [content]
                best_section = max(
                    sections,
                    key=lambda section: sum(
                        min(section.casefold().count(term), 8)
                        + (4 if term in section.split("\n", 1)[0].casefold() else 0)
                        for term in terms
                    ),
                    default="",
                )
                lowered = best_section.casefold()
                matched = [term for term in terms if term in lowered or term in relative_name]
                if len(matched) < required_matches or any(term not in matched for term in numeric_terms):
                    continue
                score = sum(min(lowered.count(term), 8) + (4 if term in title else 0)
                            + (2 if term in relative_name else 0) for term in terms)
                content_matches = [term for term in terms if term in lowered]
                if len(content_matches) >= required_matches and all(term in lowered for term in numeric_terms):
                    score += 30
                if is_theory and not numeric_terms:
                    score += 100
                positions = [lowered.find(term) for term in terms if term in lowered]
                start = 0 if is_theory else max(0, min(positions) - 130) if positions else 0
                excerpt = " ".join(best_section[start:start + 650].split()) or "Совпадение в названии файла или папки; текст не извлечён."
                candidates.append((score, {"path": str(path), "excerpt": excerpt, "kind": "document"}))
            if scanned >= MAX_FILES or visited >= MAX_ENTRIES:
                break
    if stats is not None:
        stats.update(documents_read=scanned, entries_seen=visited_total, truncated=truncated)
    candidates.sort(key=lambda pair: (-pair[0], pair[1]["path"].casefold()))
    return [item for _, item in candidates[:max(1, min(limit, 12))]]
