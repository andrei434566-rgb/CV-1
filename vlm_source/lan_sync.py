"""Exchange core source files through a private LAN folder, without training on upload.

The shared folder is the transport. Only the training computer runs ``ingest``.
All incoming files are treated as unreviewed candidates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import sys
import uuid


SCHEMA = "coresed-lan-sync-v1"
ALLOWED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp",
    ".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".pdf", ".docx",
    ".json", ".jsonl", ".txt", ".zip", ".7z",
}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "model_cache", "outputs"}
COPY_BLOCK = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
        raise ValueError("Имя ПК: 1–40 латинских букв, цифр, _ или -")
    return value


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def relative_file(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Небезопасный относительный путь в манифесте")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("Небезопасный относительный путь в манифесте")
    return Path(*pure.parts)


def digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(COPY_BLOCK):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def copy_and_digest(source: Path, target: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, target.open("xb") as output_stream:
        while block := input_stream.read(COPY_BLOCK):
            output_stream.write(block)
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def hub_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    marker = root / "hub.json"
    if not marker.is_file() or read_json(marker).get("schema") != SCHEMA:
        raise ValueError(f"Хранилище не инициализировано: {root}")
    return root


def initialize(value: str) -> None:
    root = Path(value).expanduser().resolve()
    marker = root / "hub.json"
    if marker.exists():
        if read_json(marker).get("schema") != SCHEMA:
            raise ValueError("В папке уже есть другое хранилище")
        print(f"Хранилище уже готово: {root}")
        return
    for name in ("incoming", "objects", "catalog/receipts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    write_json_atomic(marker, {"schema": SCHEMA, "created_at": utc_now(),
                               "note": "Private LAN source archive; upload never approves training."})
    print(f"Хранилище создано: {root}")


def source_files(root: Path):
    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, child_dirs, names in os.walk(root, followlinks=False, onerror=raise_walk_error):
        child_dirs[:] = sorted(d for d in child_dirs if d.lower() not in SKIP_DIRS
                               and not d.startswith(".") and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if name.startswith("~$") or path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() in ALLOWED_SUFFIXES:
                yield path


def submit(args: argparse.Namespace) -> None:
    root = hub_root(args.hub)
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Папка источника не найдена: {source}")
    if is_within(source, root) or is_within(root, source):
        raise ValueError("Источник и хранилище не должны находиться друг внутри друга")
    computer = safe_id(args.pc)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    device_inbox = root / "incoming" / computer
    device_inbox.mkdir(parents=True, exist_ok=True)
    staging = device_inbox / f".upload-{batch_id}"
    final = device_inbox / batch_id
    staging.mkdir()
    records = []
    try:
        for path in source_files(source):
            relative = path.relative_to(source)
            before = path.stat()
            target = staging / "files" / relative
            digest, size = copy_and_digest(path, target)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
                raise RuntimeError(f"Файл изменился во время копирования: {path}")
            records.append({"path": relative.as_posix(), "size": size, "sha256": digest})
        if not records:
            raise ValueError("Не найдено фото, таблиц или документов поддерживаемых форматов")
        manifest = {
            "schema": SCHEMA, "batch_id": batch_id, "computer": computer,
            "source_id": args.source_id or source.name, "source_kind": args.kind,
            "source_path_on_sender": str(source), "created_at": utc_now(),
            "training_approved": False, "file_count": len(records),
            "total_bytes": sum(row["size"] for row in records), "files": records,
        }
        write_json_atomic(staging / "manifest.json", manifest)
        if not is_within(staging, root) or not is_within(final, root):
            raise RuntimeError("Путь пакета вышел за границы хранилища")
        os.replace(staging, final)
    except Exception:
        if staging.exists() and is_within(staging, root):
            shutil.rmtree(staging)
        raise
    print(json.dumps({"batch_id": batch_id, "files": len(records),
                      "bytes": manifest["total_bytes"], "package": str(final)}, ensure_ascii=False))


def ingest_one(root: Path, batch_dir: Path, computer: str) -> dict:
    manifest = read_json(batch_dir / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("computer") != computer or manifest.get("batch_id") != batch_dir.name:
        raise ValueError(f"Неверный манифест: {batch_dir}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files or len(files) != manifest.get("file_count"):
        raise ValueError(f"Неверный список файлов: {batch_dir}")
    receipt = root / "catalog" / "receipts" / computer / f"{batch_dir.name}.json"
    if receipt.exists():
        return {"batch_id": batch_dir.name, "status": "already_ingested"}
    seen = set()
    verified = []
    new_objects = 0
    for row in files:
        relative = relative_file(row["path"])
        if str(relative).lower() in seen:
            raise ValueError(f"Повтор пути внутри пакета: {relative}")
        seen.add(str(relative).lower())
        source = batch_dir / "files" / relative
        if source.is_symlink() or not source.is_file() or not is_within(source, batch_dir / "files"):
            raise ValueError(f"Отсутствует или небезопасен файл: {relative}")
        sha256, size = digest_file(source)
        if sha256 != row.get("sha256") or size != row.get("size"):
            raise ValueError(f"Не совпала контрольная сумма: {relative}")
        object_path = root / "objects" / sha256[:2] / sha256
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if not is_within(object_path, root / "objects"):
            raise ValueError("Путь объекта вышел за границы хранилища")
        if object_path.exists():
            existing_sha, existing_size = digest_file(object_path)
            if (existing_sha, existing_size) != (sha256, size):
                raise ValueError(f"Повреждён объект в хранилище: {object_path}")
        else:
            temporary = object_path.with_name(f".{sha256}.{uuid.uuid4().hex}.tmp")
            copied_sha, copied_size = copy_and_digest(source, temporary)
            if (copied_sha, copied_size) != (sha256, size):
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Файл изменился во время импорта: {relative}")
            os.replace(temporary, object_path)
            new_objects += 1
        verified.append({"path": row["path"], "size": size, "sha256": sha256,
                         "object": str(object_path.relative_to(root).as_posix())})
    write_json_atomic(receipt, {
        "schema": SCHEMA, "batch_id": batch_dir.name, "computer": computer,
        "source_id": manifest.get("source_id"), "source_kind": manifest.get("source_kind"),
        "source_path_on_sender": manifest.get("source_path_on_sender"),
        "created_at": manifest.get("created_at"), "ingested_at": utc_now(),
        "training_approved": False, "review_status": "unreviewed_candidate",
        "files": verified,
    })
    return {"batch_id": batch_dir.name, "status": "ingested", "files": len(verified),
            "new_objects": new_objects}


def ingest(args: argparse.Namespace) -> None:
    root = hub_root(args.hub)
    results = []
    for device in sorted((root / "incoming").iterdir()):
        if not device.is_dir() or device.is_symlink() or device.name.startswith("."):
            continue
        computer = safe_id(device.name)
        for batch in sorted(device.iterdir()):
            if not batch.is_dir() or batch.is_symlink() or batch.name.startswith("."):
                continue
            try:
                results.append(ingest_one(root, batch, computer))
            except (ValueError, OSError, KeyError, TypeError) as error:
                results.append({"batch_id": batch.name, "status": "rejected", "reason": str(error)})
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(row["status"] == "rejected" for row in results):
        raise ValueError("Часть пакетов отклонена; смотрите причины выше")


def status(args: argparse.Namespace) -> None:
    root = hub_root(args.hub)
    receipts = list((root / "catalog" / "receipts").glob("*/*.json"))
    objects = [path for path in (root / "objects").glob("*/*") if path.is_file()]
    incoming = [path for path in (root / "incoming").glob("*/*")
                if path.is_dir() and not path.name.startswith(".")]
    print(json.dumps({"hub": str(root), "completed_uploads": len(incoming),
                      "ingested_batches": len(receipts),
                      "waiting_for_ingest": max(0, len(incoming) - len(receipts)),
                      "unique_objects": len(objects),
                      "stored_bytes": sum(path.stat().st_size for path in objects),
                      "automatically_approved_for_training": 0}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser("init", help="Создать центральное хранилище")
    init_parser.add_argument("--hub", required=True)
    init_parser.set_defaults(action=lambda args: initialize(args.hub))
    submit_parser = commands.add_parser("submit", help="Отправить фото, Excel и документы из папки")
    submit_parser.add_argument("--hub", required=True)
    submit_parser.add_argument("--source", required=True)
    submit_parser.add_argument("--pc", default=re.sub(r"[^A-Za-z0-9_-]", "_", socket.gethostname())[:40])
    submit_parser.add_argument("--source-id", default="")
    submit_parser.add_argument("--kind", choices=("raw", "app_sessions", "prepared"), default="raw")
    submit_parser.set_defaults(action=submit)
    ingest_parser = commands.add_parser("ingest", help="Проверить и принять готовые пакеты")
    ingest_parser.add_argument("--hub", required=True)
    ingest_parser.set_defaults(action=ingest)
    status_parser = commands.add_parser("status", help="Показать объём принятых данных")
    status_parser.add_argument("--hub", required=True)
    status_parser.set_defaults(action=status)
    args = parser.parse_args()
    try:
        args.action(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
