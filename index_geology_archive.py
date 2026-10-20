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
DEPTH_RE = re.compile(r"(?<!\d)(\d{2,5}[.,]\d{1,3})\s*[-–—_]\s*(\d{2,5}[.,]\d{1,3})(?!\d)")
TOP_HINTS = ("от", "верх", "начал", "top", "from", "глубина от")
BASE_HINTS = ("до", "низ", "конец", "bottom", "to", "глубина до")
TEXT_HINTS = ("описан", "литолог", "фаци", "пород", "характер", "description")


def filename_interval(name: str):
    match = DEPTH_RE.search(name)
    if match:
        a, b = (float(v.replace(",", ".")) for v in match.groups())
        return [a, b] if a < b else None
    return None


def numeric(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        value = value.strip().replace(" ", "").replace(",", ".")
        if re.fullmatch(r"\d{2,5}\.\d{1,3}", value):
            return float(value)
    return None


def excel_intervals(path: Path):
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Install openpyxl: python -m pip install openpyxl") from exc
    result = []
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
        for sheet in workbook:
            iterator = sheet.iter_rows(values_only=True)
            for row_index, header in enumerate(iterator, start=1):
                labels = [str(v or "").strip().casefold() for v in header]
                top_cols = [i for i, h in enumerate(labels) if any(t in h for t in TOP_HINTS)]
                base_cols = [i for i, h in enumerate(labels) if any(t in h for t in BASE_HINTS)]
                desc_cols = [i for i, h in enumerate(labels) if any(t in h for t in TEXT_HINTS)]
                if not (top_cols and base_cols and desc_cols):
                    if row_index >= 35:
                        break
                    continue
                ti, bi, di = top_cols[0], base_cols[0], desc_cols[0]
                for number, row in enumerate(iterator, start=row_index + 1):
                    top = numeric(row[ti]) if len(row) > ti else None
                    base = numeric(row[bi]) if len(row) > bi else None
                    description = str(row[di] or "").strip() if len(row) > di else ""
                    if top is None or base is None or not (top < base) or not description:
                        continue
                    result.append(dict(file=str(path), sheet=sheet.title, row=number,
                                       top=top, bottom=base, description=description))
                break
        workbook.close()
    except Exception as exc:
        return [], str(exc)
    return result, None


from geology_excel_parser import excel_intervals as excel_intervals\n\n\ndef scan(root: Path, output: Path):
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    photos, intervals, documents, errors = [], [], [], []
    wells = [p for p in root.iterdir() if p.is_dir() and p.resolve() != output
             and p.name.casefold() not in THEORY_NAMES and not p.name.startswith("_index_cv1")]
    for folder in sorted(wells):
        for file in folder.rglob("*"):
            if not file.is_file() or output in file.resolve().parents:
                continue
            suffix = file.suffix.casefold()
            if suffix in IMAGE_EXT:
                photos.append(dict(well=folder.name, file=str(file), interval=filename_interval(file.stem)))
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
            pairs.append(dict(well=photo["well"], photo=photo["file"], status="missing_photo_depth",
                              confidence="none", matches=[]))
            continue
        matches = []
        for row in by_well.get(photo["well"], []):
            overlap = max(0.0, min(span[1], row["bottom"]) - max(span[0], row["top"]))
            if overlap > 0:
                matches.append(dict(excel=row["file"], sheet=row["sheet"], row=row["row"],
                                    top=row["top"], bottom=row["bottom"],
                                    overlap_fraction=round(overlap / (span[1] - span[0]), 4),
                                    description=row["description"]))
        pairs.append(dict(well=photo["well"], photo=photo["file"], photo_depth=span,
                          status="review_required" if matches else "no_matching_description",
                          confidence="candidate" if matches else "none", matches=matches))
    output.mkdir(parents=True, exist_ok=True)
    data = dict(source=str(root), wells=[p.name for p in wells], photos=photos,
                descriptions=intervals, theory=documents, pairs=pairs, errors=errors)
    (output / "archive_index.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "pair_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Скважина", "Фото", "Глубина от", "Глубина до", "Статус", "Совпадений Excel"])
        for item in pairs:
            span = item.get("photo_depth") or ["", ""]
            writer.writerow([item["well"], item["photo"], *span, item["status"], len(item["matches"])])
    return dict(wells=len(wells), photos=len(photos), descriptions=len(intervals),
                documents=len(documents), candidate_photos=sum(bool(p["matches"]) for p in pairs),
                errors=len(errors), output=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV-1 geological archive indexer")
    parser.add_argument("folder", type=Path, help="Archive root with one folder per well")
    parser.add_argument("--output", type=Path, help="Output directory (default: _index_cv1 inside archive)")
    args = parser.parse_args()
    print(json.dumps(scan(args.folder, args.output or args.folder / "_index_cv1"),
                     ensure_ascii=False, indent=2))
