"""Numbered photographs paired with verbatim descriptions from one DOCX."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
_HEADER = re.compile(
    r"^\s*(?:фото(?:графия)?)\s*(?:№\s*|#\s*)?"
    r"(?P<number>\d+)(?:\.(?:jpe?g|png|tiff?|bmp|webp))?"
    r"(?=$|[\s:;.—–-])\s*[:;.—–-]?\s*(?P<text>.*)$", re.IGNORECASE,
)


def _number(value: str) -> str:
    """0001 and 1 identify the same photo; 00010 and 0001 do not."""
    return value.lstrip("0") or "0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_lines(container):
    # Preserve Word order, including tables pasted from Excel and nested tables.
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            yield block.text
        elif isinstance(block, Table):
            seen_cells = set()
            for row in block.rows:
                for cell in row.cells:
                    if cell._tc not in seen_cells:  # merged cells appear repeatedly
                        seen_cells.add(cell._tc)
                        yield from _document_lines(cell)


def read_numbered_descriptions(source: Path) -> list[tuple[str, str]]:
    from docx import Document

    source = Path(source)
    if source.suffix.casefold() != ".docx":
        raise ValueError("Нужен Word в формате .docx. Старый .doc сохраните как .docx.")
    descriptions: list[tuple[str, str]] = []
    number = None
    paragraphs: list[str] = []
    for paragraph in _document_lines(Document(source)):
        # A heading can follow a manual line break within a Word paragraph.
        for line in paragraph.split("\n"):
            heading = _HEADER.fullmatch(line)
            if heading:
                if number is not None:
                    descriptions.append((number, "\n".join(paragraphs).strip()))
                number = heading["number"]
                paragraphs = [heading["text"]] if heading["text"] else []
            elif number is not None:
                paragraphs.append(line)
    if number is not None:
        descriptions.append((number, "\n".join(paragraphs).strip()))
    if not descriptions:
        raise ValueError("В Word не найдены разделы «Фото 0001». Пишите номер на отдельной строке перед описанием.")
    return descriptions


@dataclass(frozen=True)
class WordPhotoPair:
    photo_id: str
    description: str
    images: tuple[Path, ...]
    status: str
    message: str
    image_sha256: str = ""


@dataclass(frozen=True)
class WordPhotoInspection:
    photos_folder: Path
    document: Path
    document_sha256: str
    pairs: tuple[WordPhotoPair, ...]

    @property
    def ready_pairs(self) -> tuple[WordPhotoPair, ...]:
        return tuple(pair for pair in self.pairs if pair.status == "ready")


def inspect_word_photo_pairs(photos_folder: Path, document: Path) -> WordPhotoInspection:
    folder, document = Path(photos_folder).resolve(), Path(document).resolve()
    if not folder.is_dir():
        raise ValueError("Выберите существующую папку с фотографиями.")
    document_hash = _sha256(document)
    descriptions = read_numbered_descriptions(document)
    if _sha256(document) != document_hash:
        raise ValueError("Word изменился во время чтения. Повторите проверку.")
    images = defaultdict(list)
    unnumbered = []
    for image in sorted(folder.rglob("*")):
        if image.is_file() and image.suffix.casefold() in IMAGE_EXTENSIONS:
            if image.stem.isdecimal():
                images[_number(image.stem)].append(image)
            else:
                unnumbered.append(image)
    counts = Counter(_number(number) for number, _ in descriptions)
    pairs = []
    for number, text in descriptions:
        candidates = tuple(images.get(_number(number), []))
        problems = []
        if counts[_number(number)] > 1:
            problems.append("Номер повторяется в Word")
        if not text:
            problems.append("Пустое описание")
        if not candidates:
            problems.append("Фотография не найдена")
        elif len(candidates) > 1:
            problems.append("Несколько фотографий с этим номером")
        image_hash = ""
        if len(candidates) == 1:
            try:
                with Image.open(candidates[0]) as image:
                    image.verify()
                image_hash = _sha256(candidates[0])
            except (OSError, ValueError, Image.DecompressionBombError):
                problems.append("Фото повреждено или не читается")
        pairs.append(WordPhotoPair(number, text, candidates, "error" if problems else "ready",
                                   "; ".join(problems) if problems else "Готово", image_hash))
    for number, candidates in images.items():
        if number not in counts:
            pairs.append(WordPhotoPair(candidates[0].stem, "", tuple(candidates), "unmatched", "Нет описания в Word"))
    for image in unnumbered:
        pairs.append(WordPhotoPair(image.stem, "", (image,), "unmatched", "Имя фото должно быть номером, например 0001.jpg"))
    return WordPhotoInspection(folder, document, document_hash, tuple(pairs))


def export_word_photo_pairs(inspection: WordPhotoInspection, destination: Path, *, approved: bool = False) -> dict:
    """Copy full originals and create a portable, locally trainable manifest."""
    if not approved:
        raise ValueError("Подтвердите, что описания и соответствие номеров проверены.")
    if not inspection.ready_pairs:
        raise ValueError("Нет готовых пар. Исправьте ошибки в Word и именах фото.")
    if _sha256(inspection.document) != inspection.document_sha256:
        raise ValueError("Word изменился после проверки. Нажмите «Проверить пары» повторно.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    image_dir = destination / "images"
    image_dir.mkdir()
    shutil.copyfile(inspection.document, destination / "descriptions.docx")
    if _sha256(destination / "descriptions.docx") != inspection.document_sha256:
        raise ValueError("Word изменился во время копирования. Повторите проверку.")
    rows = []
    for pair in inspection.ready_pairs:
        original = pair.images[0]
        name = f"{pair.photo_id}{original.suffix.casefold()}"
        copied = image_dir / name
        shutil.copyfile(original, copied)
        if _sha256(copied) != pair.image_sha256:
            raise ValueError(f"Фото {pair.photo_id} изменилось после проверки. Проверьте пары заново.")
        rows.append({
            "image": f"images/{name}",
            "conversations": [
                {"from": "human", "value": "<image>Опиши керн на русском языке по видимым признакам. Отделяй наблюдения от интерпретации."},
                {"from": "gpt", "value": pair.description},
            ],
            "metadata": {
                "training_approved": True, "geology_approved": True,
                "record_type": "image_text", "label_provenance": "expert_word_photo_number_reviewed",
                "photo_id": pair.photo_id, "source_photo": str(original),
                "source_document": str(inspection.document),
                "source_document_sha256": inspection.document_sha256,
                "image_sha256": pair.image_sha256,
                "dataset": f"word_photo_{inspection.document_sha256[:16]}",
            },
        })
    report = {
        "schema": "cv1-word-photo-pairs-v1", "created_at": datetime.now().isoformat(timespec="seconds"),
        "approved_pairs": len(rows), "document_sha256": inspection.document_sha256,
        "skipped": [{"photo_id": pair.photo_id, "reason": pair.message}
                    for pair in inspection.pairs if pair.status != "ready"],
        "note": "Полные фотографии и исходные описания без сокращений. Это пары для VLM; контуров для обучения YOLO здесь нет. Контрольные скважины подготовьте отдельно.",
    }
    (destination / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Publish approved rows last, after every image was checked and copied.
    manifest = destination / "word_photo_approved.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"manifest": manifest, "count": len(rows), "skipped": len(report["skipped"])}
