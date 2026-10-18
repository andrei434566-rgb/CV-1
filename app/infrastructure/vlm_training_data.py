"""Export only geologist-approved image descriptions and explicit theory Q&A."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPolygonF


def reviewed_pair_counts(records) -> tuple[int, int]:
    reviewed = usable = 0
    for record in records:
        for detection in record.detections:
            if not detection.training_ready:
                continue
            reviewed += 1
            attributes = detection.attributes or {}
            if (attributes.get("Краткое описание") or attributes.get("Литологическое описание (16 параметров)")) and str(attributes.get("Источник описания", "")).startswith("Excel:") and len(detection.polygon) >= 3:
                usable += 1
    return reviewed, usable


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def export_reviewed_pairs(records, destination: Path) -> dict:
    """Save masked interval crops, direct Excel text, and auditable provenance."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Учебная папка уже существует: {destination}")
    crops = destination / "crops"
    crops.mkdir(parents=True)
    rows = []
    skipped = []
    for record in records:
        if record.pixmap.isNull():
            continue
        image = record.pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
        for index, detection in enumerate(record.detections):
            if not detection.training_ready:
                continue
            attrs = detection.attributes or {}
            description = str(attrs.get("Краткое описание") or attrs.get("Литологическое описание (16 параметров)") or "").strip()
            if not description or not str(attrs.get("Источник описания", "")).startswith("Excel:") or len(detection.polygon) < 3:
                skipped.append({"photo": record.path, "layer": index, "reason": "Нет исходного описания Excel или корректного контура"})
                continue
            polygon = QPolygonF(detection.polygon)
            rect = polygon.boundingRect().toAlignedRect().intersected(image.rect())
            if rect.width() < 8 or rect.height() < 8:
                skipped.append({"photo": record.path, "layer": index, "reason": "Слишком маленький контур"})
                continue
            crop = QImage(rect.size(), QImage.Format.Format_RGB32)
            crop.fill(Qt.GlobalColor.white)
            painter = QPainter(crop)
            path = QPainterPath()
            path.addPolygon(polygon.translated(-rect.x(), -rect.y()))
            painter.setClipPath(path)
            painter.drawImage(-rect.x(), -rect.y(), image)
            painter.end()
            name = f"pair_{len(rows)+1:06d}.jpg"
            if not crop.save(str(crops / name), "JPG", 94):
                raise OSError(f"Не удалось сохранить вырезку: {name}")
            rows.append({
                "image": f"crops/{name}",
                "conversations": [
                    {"from": "human", "value": "<image>Опиши видимые признаки породы на русском языке. Не выдумывай невидимые детали."},
                    {"from": "gpt", "value": description},
                ],
                "metadata": {
                    "training_approved": True,
                    "geology_approved": True,
                    "record_type": "image_text",
                    "label_provenance": "source_direct_excel_reviewed_mask",
                    "source_photo": str(record.path),
                    "source_well": record.well_name,
                    "source_row": attrs.get("Источник описания", ""),
                    "facies": detection.label,
                    "depth_top": detection.depth_from,
                    "depth_base": detection.depth_to,
                },
            })
    if not rows:
        raise ValueError("Нет проверенных контуров с исходным описанием Excel. Подтвердите слои в карточках фаций.")
    manifest = destination / "photo_excel_approved.jsonl"
    _write_jsonl(manifest, rows)
    (destination / "export_report.json").write_text(json.dumps({
        "schema": "kern-vlm-photo-excel-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "approved_pairs": len(rows), "skipped": skipped,
        "warning": "Проверка контура геологом является условием экспорта. Контрольное качество оценивайте отдельно по независимым скважинам.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest, "count": len(rows), "skipped": len(skipped)}


_QUESTION = {"вопрос", "question", "запрос"}
_ANSWER = {"ответ", "answer"}


def _qa_rows(path: Path) -> list[tuple[str, str, str]]:
    result = []
    if path.suffix.casefold() == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            header = [str(value or "").strip().casefold() for value in next(rows, ())]
            q = next((i for i, h in enumerate(header) if h in _QUESTION), None)
            a = next((i for i, h in enumerate(header) if h in _ANSWER), None)
            if q is None or a is None:
                continue
            for row_number, cells in enumerate(rows, start=2):
                question = str(cells[q] or "").strip() if q < len(cells) else ""
                answer = str(cells[a] or "").strip() if a < len(cells) else ""
                if question and answer:
                    result.append((question, answer, f"{sheet.title}!{row_number}"))
        book.close()
    elif path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(2048)
            stream.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            reader = csv.DictReader(stream, dialect=dialect)
            for number, row in enumerate(reader, start=2):
                normalized = {str(k).strip().casefold(): str(v or "").strip() for k, v in row.items() if k}
                question = next((normalized[k] for k in _QUESTION if normalized.get(k)), "")
                answer = next((normalized[k] for k in _ANSWER if normalized.get(k)), "")
                if question and answer:
                    result.append((question, answer, f"строка {number}"))
    else:
        raise ValueError("Для отдельного обучения теории нужна таблица .xlsx или .csv с колонками «Вопрос» и «Ответ».")
    return result


def export_theory_qa(source: Path, destination: Path) -> dict:
    """Convert only user-authored explicit answers, never inferred geology."""
    source = Path(source)
    pairs = _qa_rows(source)
    if not pairs:
        raise ValueError("В таблице не найдены заполненные пары «Вопрос» — «Ответ».")
    records = [{
        "conversations": [
            {"from": "human", "value": question + " Отвечай по-русски; если данных недостаточно, сообщи об этом."},
            {"from": "gpt", "value": answer},
        ],
        "metadata": {
            "training_approved": True, "record_type": "text_only", "source_group": str(source),
            "source_row": address, "label_provenance": "user_supplied_theory_qa",
        },
    } for question, answer, address in pairs]
    _write_jsonl(destination, records)
    return {"manifest": destination, "count": len(records)}
