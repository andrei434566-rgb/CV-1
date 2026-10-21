"""Index a local geological archive without modifying source files.

Usage: python index_geology_archive.py "D:\\Geology archive"
Requires openpyxl for Excel parsing. Outputs JSON/CSV under an independent
_index_cv1 folder. Pairings are CANDIDATES, not verified training labels.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SHEET_EXT = {".xlsx", ".xlsm"}
DOC_EXT = {".pdf", ".docx", ".doc", ".txt", ".md"}
THEORY_NAMES = {"теория", "theory", "knowledge", "литература", "методички"}
DEPTH_RE = re.compile(r"(?<!\d)(\d{3,5}(?:[.,]\d{1,3})?)\s*[-–—_]\s*(\d{3,5}(?:[.,]\d{1,3})?)(?!\d)")
TOP_HINTS = ("от", "верх", "начал", "top", "from", "глубина от")
BASE_HINTS = ("до", "низ", "конец", "bottom", "to", "глубина до")
TEXT_HINTS = ("описан", "литолог", "фаци", "пород", "характер", "description")


MAX_BOX_INTERVAL = 5.0

def filename_interval(name: str):
    """Select a plausible core interval <=5m, not the first drilling-run span."""
    valid = []
    for match in DEPTH_RE.finditer(name):
        top, bottom = (float(v.replace(",", ".")) for v in match.groups())
        if 0 < bottom-top <= MAX_BOX_INTERVAL + 1e-7:
            valid.append([top, bottom])
    return valid[0] if len(valid) == 1 else None


def filename_interval_status(name: str):
    matches = list(DEPTH_RE.finditer(name))
    valid = []
    for m in matches:
        a,b=(float(v.replace(",", ".")) for v in m.groups())
        if 0 < b-a <= MAX_BOX_INTERVAL + 1e-7:
            valid.append([a,b])
    if len(valid) == 1:
        return "filename_short_interval"
    if len(valid) > 1:
        return "ambiguous_filename_intervals"
    return "requires_photo_depth_reading" if matches else "no_filename_interval"


def numeric(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        value = value.strip().replace(" ", "").replace(",", ".")
        if re.fullmatch(r"\d{2,5}\.\d{1,3}", value):
            return float(value)
    return None


from geology_excel_parser import excel_intervals

# Filename rule is mandatory and takes precedence over image analysis.
# Word-like abbreviations are delimited; Russian/English UV roots may be inflected.
UV_RE = re.compile(
    r"(?i)(?<![a-zа-яё0-9])(?:uv|u[._ -]?v|уф)(?![a-zа-яё0-9])"
    r"|ультра[ _.-]?фиолет[а-яё]*"
    r"|ultra[ _.-]?violet[a-z]*"
    r"|fluorescen[a-z]*|флуоресцен[а-яё]*|люминесцен[а-яё]*"
)

def uv_filename(path: Path):
    return bool(UV_RE.search(path.stem)) or any(UV_RE.search(parent.name) for parent in list(path.parents)[:2])

def likely_uv_visual(path: Path):
    """Conservative visual heuristic. Suspected UV photos require review, not deletion."""
    try:
        from PySide6.QtGui import QImageReader
        from PySide6.QtCore import QSize
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        reader.setScaledSize(QSize(96, 96))
        image = reader.read()
        if image.isNull():
            return False
        count = image.width() * image.height()
        if not count:
            return False
        colored = 0
        dark = 0
        neon_blue = 0
        for y in range(image.height()):
            for x in range(image.width()):
                c = image.pixelColor(x, y)
                h, saturation, value, _ = c.getHsv()
                if 185 <= h <= 305 and saturation >= 100 and value >= 65:
                    colored += 1
                if value <= 100:
                    dark += 1
                if 175 <= h <= 265 and saturation >= 95 and value >= 45:
                    neon_blue += 1
        return ((colored / count > 0.25 and dark / count > 0.25)
                or (neon_blue / count > 0.08 and dark / count > 0.40))
    except Exception:
        return False



def scan(root: Path, output: Path):
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    photos, intervals, documents, errors = [], [], [], []
    excluded_uv = []
    suspected_uv = []
    wells = [p for p in root.iterdir() if p.is_dir() and p.resolve() != output
             and p.name.casefold() not in THEORY_NAMES and not p.name.startswith("_index_cv1")]
    for folder in sorted(wells):
        for file in folder.rglob("*"):
            if not file.is_file() or output in file.resolve().parents:
                continue
            suffix = file.suffix.casefold()
            if suffix in IMAGE_EXT:
                if uv_filename(file):
                    excluded_uv.append(str(file))
                    continue
                if likely_uv_visual(file):
                    suspected_uv.append(str(file))
                    continue
                photos.append(dict(well=folder.name, file=str(file), interval=filename_interval(file.stem), depth_status=filename_interval_status(file.stem)))
            elif suffix in SHEET_EXT and not file.name.startswith("~$"):
                rows, error = excel_intervals(file)
                if error:
                    errors.append(dict(file=str(file), error=error))
                intervals.extend(dict(well=folder.name, **row) for row in rows)
            elif suffix in DOC_EXT:
                documents.append(dict(well=folder.name, file=str(file), type="well_document"))
    for file in root.iterdir():
        if file.is_file() and file.suffix.casefold() in DOC_EXT | SHEET_EXT:
            documents.append(dict(well=None, file=str(file), type="root_theory"))
        elif file.is_dir() and file.name.casefold() in THEORY_NAMES:
            for doc in file.rglob("*"):
                if doc.is_file() and doc.suffix.casefold() in DOC_EXT | SHEET_EXT:
                    documents.append(dict(well=None, file=str(doc), type="theory"))
    by_well = {}
    for interval in intervals:
        by_well.setdefault(interval["well"], []).append(interval)
    pairs = []
    for photo in photos:
        span = photo["interval"]
        if not span:
            pairs.append(dict(well=photo["well"], photo=photo["file"], status=photo["depth_status"],
                              confidence="none", matches=[]))
            continue
        matches = []
        for row in by_well.get(photo["well"], []):
            overlap = max(0.0, min(span[1], row["bottom"]) - max(span[0], row["top"]))
            # Facies labels require drilling depths, index/name and a
            # physically consistent layer thickness. Reject aggregate rows.
            thickness = row.get("thickness")
            depth_length = row["bottom"] - row["top"]
            # A whole-photo interval is an aggregate, not a separately mapped facies.
            # Facies names can exist even when the index is blank; do not discard them.
            is_sublayer = depth_length < (span[1] - span[0]) - 0.01
            has_label = bool(str(row.get("facies_index") or "").strip() or
                             str(row.get("facies") or "").strip())
            consistent = (thickness is None or
                          abs(float(thickness) - depth_length) <= max(0.05, depth_length * 0.05))
            if overlap > 0 and is_sublayer and has_label and consistent:
                matches.append(dict(excel=row["file"], sheet=row["sheet"], row=row["row"],
                                    top=row["top"], bottom=row["bottom"],
                                    overlap_fraction=round(overlap / (span[1] - span[0]), 4),
                                    description=row["description"], facies=row.get("facies", ""), lithology=row.get("lithology", ""),
                                    facies_index=row.get("facies_index", ""), facies_code=row.get("facies_code", ""),
                                    thickness=row.get("thickness")) )
        pairs.append(dict(well=photo["well"], photo=photo["file"], photo_depth=span, depth_source="filename_short_interval",
                          status="review_required" if matches else "no_matching_description",
                          confidence="candidate" if matches else "none", matches=matches))
    output.mkdir(parents=True, exist_ok=True)
    data = dict(source=str(root), wells=[p.name for p in wells], photos=photos,
                descriptions=intervals, theory=documents, pairs=pairs, errors=errors, excluded_uv=excluded_uv, suspected_uv=suspected_uv)
    (output / "archive_index.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "pair_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Скважина", "Фото", "Глубина от", "Глубина до", "Статус", "Совпадений Excel"])
        for item in pairs:
            span = item.get("photo_depth") or ["", ""]
            writer.writerow([item["well"], item["photo"], *span, item["status"], len(item["matches"])])
    return dict(wells=len(wells), photos=len(photos), descriptions=len(intervals),
                documents=len(documents), candidate_photos=sum(bool(p["matches"]) for p in pairs),
                errors=len(errors), excluded_uv=len(excluded_uv), suspected_uv=len(suspected_uv), output=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV-1 geological archive indexer")
    parser.add_argument("folder", type=Path, help="Archive root with one folder per well")
    parser.add_argument("--output", type=Path, help="Output directory (default: _index_cv1 inside archive)")
    args = parser.parse_args()
    print(json.dumps(scan(args.folder, args.output or args.folder / "_index_cv1"),
                     ensure_ascii=False, indent=2))
