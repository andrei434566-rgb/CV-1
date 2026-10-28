"""Import the user's local documents into a persistent, searchable text index.

Import provides retrieval context; it does not train model weights. Source files
are treated as untrusted data, never executed, and kept in local user storage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from xml.etree import ElementTree
import zipfile

from PySide6.QtCore import QThread, Signal

MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TEXT_CHARS = 10_000_000
MAX_PAGES = 1000
CHUNK_CHARS = 8000
SUPPORTED = {".txt", ".md", ".csv", ".tsv", ".pdf", ".docx", ".json", ".jsonl"}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(path: Path) -> tuple[str, list[str]]:
    suffix = path.suffix.casefold()
    warnings: list[str] = []
    if suffix in {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl"}:
        data = path.read_bytes()
        try:
            value = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            value = data.decode("cp1251", errors="replace")
        if suffix in {".json", ".jsonl"}:
            from app.infrastructure.local_knowledge_search import _document_text
            if not _document_text(path):
                raise ValueError("Файл исключён из справочного поиска или не содержит допустимый текст")
    elif suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > 30 * 1024 * 1024:
                raise ValueError("Слишком большой текст документа DOCX")
            document = ElementTree.fromstring(archive.read(info))
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        value = "\n".join("".join(node.text or "" for node in para.iter(ns + "t"))
                          for para in document.iter(ns + "p"))
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            import sys
            vendor = Path(__file__).resolve().parents[2] / "third_party"
            if not (vendor / "pypdf").is_dir():
                raise ValueError("Для чтения PDF нужен локальный пакет pypdf")
            sys.path.insert(0, str(vendor))
            from pypdf import PdfReader
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise ValueError("PDF защищён паролем")
        count = len(reader.pages)
        pages = []
        size = 0
        blank = 0
        for index in range(min(count, MAX_PAGES)):
            content = reader.pages[index].extract_text() or ""
            blank += int(not content.strip())
            pages.append(f"Страница {index + 1}\n{content}")
            size += len(content)
            if size >= MAX_TEXT_CHARS:
                warnings.append("Прочитана только часть документа: превышен лимит текста.")
                break
        value = "\n\n".join(pages)
        if blank:
            warnings.append(f"На {blank} страницах нет извлекаемого текста; изображения и сканы без OCR не прочитаны.")
        if count > MAX_PAGES:
            warnings.append(f"Прочитаны первые {MAX_PAGES} страниц из {count}.")
        if not any(page.split("\n", 1)[-1].strip() for page in pages):
            raise ValueError("В PDF не найден текст: для сканов нужно распознавание OCR")
    else:
        raise ValueError("Формат документа не поддерживается")
    if len(value) > MAX_TEXT_CHARS:
        warnings.append("Прочитана только часть документа: превышен лимит текста.")
        value = value[:MAX_TEXT_CHARS]
    if not value.strip():
        raise ValueError("В документе не найден читаемый текст")
    return value, warnings


def import_theory_document(source: str | Path, root: str | Path) -> dict:
    source, root = Path(source), Path(root)
    if source.suffix.casefold() not in SUPPORTED:
        raise ValueError("Формат документа не поддерживается")
    if not source.is_file():
        raise ValueError("Файл не найден")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Файл больше 100 МБ; разделите его на части")
    if source.suffix.casefold() in {".json", ".jsonl"}:
        # Preserve the original-name and parent-directory quarantine rules
        # before making the stable copy under a generic staging filename.
        from app.infrastructure.local_knowledge_search import _document_text
        if not _document_text(source):
            raise ValueError("Файл исключён из справочного поиска или не содержит допустимый текст")
    root.mkdir(parents=True, exist_ok=True)
    staging_root = root / ".theory_staging"
    staging_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="document-", dir=staging_root)).resolve()
    if stage.parent != staging_root.resolve():
        raise ValueError("Недопустимая папка временных данных")
    try:
        digest = _hash(source)
        stable = stage / ("source" + source.suffix.casefold())
        shutil.copyfile(source, stable)
        if _hash(stable) != digest or _hash(source) != digest:
            raise ValueError("Файл изменился во время копирования; повторите загрузку")
        text, warnings = _extract(stable)
        if _hash(source) != digest:
            raise ValueError("Файл изменился во время чтения; повторите загрузку")
        chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
        staged_index = stage / "index"
        staged_index.mkdir()
        for number, chunk in enumerate(chunks, 1):
            target = staged_index / f"{digest}_{number:04}.txt"
            target.write_text(f"Документ: {source.name}. Фрагмент {number}/{len(chunks)}.\n\n{chunk}", encoding="utf-8")
        metadata = {"name": source.name, "sha256": digest, "chunks": len(chunks),
                    "characters": len(text), "warnings": warnings, "weights_trained": False}
        (staged_index / ".complete.json").write_text(json.dumps({"schema": "qwen_theory_chat_index_v1", **metadata},
                                                                ensure_ascii=False), encoding="utf-8")
        index = root / "theory_index"
        originals = root / "theory_originals"
        originals.mkdir(parents=True, exist_ok=True)
        if originals.resolve().parent != root.resolve():
            raise ValueError("Недопустимая папка исходных документов")
        original = originals / (digest + source.suffix.casefold())
        if not original.exists():
            os.replace(stable, original)
        elif _hash(original) != digest:
            raise ValueError("Сохранённая копия повреждена; требуется проверка специалистом")
        meta = originals / (digest + ".json")
        meta_stage = stage / "metadata.json"
        meta_stage.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(meta_stage, meta)
        index.mkdir(parents=True, exist_ok=True)
        if index.resolve().parent != root.resolve():
            raise ValueError("Недопустимая папка индекса")
        final_index = index / digest
        if final_index.resolve().parent != index.resolve():
            raise ValueError("Недопустимая папка документа")
        if final_index.exists():
            try:
                previous = json.loads((final_index / ".complete.json").read_text(encoding="utf-8"))
                existing_chunks = list(final_index.glob(f"{digest}_*.txt"))
                if previous.get("sha256") != digest or len(existing_chunks) != previous.get("chunks"):
                    raise ValueError("Индекс документа неполон; требуется проверка")
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError("Сохранённый индекс документа повреждён; требуется проверка") from exc
        else:
            # Publish the complete directory in one filesystem operation.
            os.rename(staged_index, final_index)
        return {**metadata, "index_folder": str(index), "original": str(original)}
    finally:
        # Only the newly-created per-import staging folder can be removed.
        if stage.parent == staging_root.resolve() and stage.exists():
            shutil.rmtree(stage)


class TheoryImportThread(QThread):
    document_result = Signal(object)

    def __init__(self, files: list[str], root: Path, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.root = root

    def run(self) -> None:
        for filename in self.files:
            try:
                self.document_result.emit({"ok": True, **import_theory_document(filename, self.root)})
            except Exception as exc:
                self.document_result.emit({"ok": False, "name": Path(filename).name, "error": str(exc)})
