"""Versioned Windows-share transport. Inference always uses verified local files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from uuid import uuid4

SCHEMA = "cv1-model-registry-v1"
PACKET_SCHEMA = "cv1-reviewed-packet-v1"
ROLES = {"geometry", "columns", "vlm_base", "vlm_adapter"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,99}", value) or value in {".", ".."}:
        raise ValueError("Недопустимый идентификатор версии или модели")
    return value


def relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Небезопасный путь в реестре")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {".", ".."} for part in pure.parts):
        raise ValueError("Небезопасный путь в реестре")
    return Path(*pure.parts)


def inside(root: Path, value: str) -> Path:
    path = root / relative(value)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Путь выходит за пределы хранилища")
    return path


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Ожидался JSON-объект")
    return data


def validate_registry(data: dict) -> dict:
    if data.get("schema") != SCHEMA:
        raise ValueError("Неверный формат реестра моделей")
    identifier(data.get("release_id", ""))
    models = data.get("models")
    if not isinstance(models, list) or not models or len(models) > 20:
        raise ValueError("Реестр не содержит моделей")
    ids, roles = set(), set()
    for model in models:
        model_id = identifier(model["id"])
        identifier(model["version"])
        role = model.get("role")
        if model_id in ids or role in roles or role not in ROLES:
            raise ValueError("Повторяющаяся модель или неизвестная роль")
        ids.add(model_id)
        roles.add(role)
        if model.get("approval", {}).get("approved") is not True:
            raise ValueError("Клиент не принимает неопубликованного кандидата")
        files = model.get("files")
        if not isinstance(files, list) or not files or len(files) > 10000:
            raise ValueError("Пустой или слишком большой список файлов модели")
        paths = set()
        for item in files:
            relative(item["path"])
            key = item["path"].casefold()
            if key in paths or not isinstance(item["size"], int) or item["size"] < 0 or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
                raise ValueError("Неверный файл модели в реестре")
            paths.add(key)
        if model["entrypoint"].casefold() not in paths:
            raise ValueError("Точка входа модели не входит в список файлов")
    base = next((model for model in models if model["role"] == "vlm_base"), None)
    adapter = next((model for model in models if model["role"] == "vlm_adapter"), None)
    if adapter and (not base or adapter.get("base_model_id") != base["id"] or adapter.get("base_version") != base["version"]):
        raise ValueError("LoRA несовместима с базовой VLM в реестре")
    return data


def initialize_hub(root: Path):
    root = Path(root)
    for folder in ("models", "registry/history", "training/incoming", "training/approved", "training/datasets", "reports/validation"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    if not (root / "hub.json").exists():
        atomic_json(root / "hub.json", {"schema": "cv1-network-hub-v1", "created_at": utc_now()})


def verify_files(root: Path, files: list[dict]) -> bool:
    for item in files:
        path = inside(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            return False
    return True


def sync_models(hub: Path, cache: Path) -> dict:
    hub, cache = Path(hub), Path(cache)
    registry = validate_registry(read_json(hub / "registry/models.json"))
    registry_hash = hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()
    active_path = cache / "active.json"
    previous = read_json(active_path) if active_path.is_file() else None
    local_models = []
    for model in registry["models"]:
        target = cache / "models" / model["id"] / model["version"]
        complete = target / ".complete.json"
        if complete.is_file() and read_json(complete).get("files") != model["files"]:
            raise ValueError("Сервер изменил содержимое существующей версии. Требуется новый номер версии")
        if not verify_files(target, model["files"]):
            source = hub / "models" / model["id"] / model["version"]
            required = sum(item["size"] for item in model["files"])
            cache.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(cache).free < required + 128 * 1024 * 1024:
                raise OSError("Недостаточно места для новой модели; предыдущая версия сохранена")
            stage = target.with_name(f".{target.name}.{uuid4().hex}.download")
            stage.mkdir(parents=True)
            for item in model["files"]:
                origin, destination = inside(source, item["path"]), inside(stage, item["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, destination)
                if destination.stat().st_size != item["size"] or digest(destination) != item["sha256"]:
                    raise ValueError("Контрольная сумма модели не совпала; активная версия сохранена")
            atomic_json(stage / ".complete.json", model)
            if target.exists():
                # A damaged immutable cache stays available for diagnosis.
                target.rename(target.with_name(f"{target.name}.damaged-{uuid4().hex[:6]}"))
            stage.rename(target)
        entry = inside(target, model["entrypoint"])
        local_models.append({**model, "local_folder": str(target.resolve()), "local_entrypoint": str(entry.resolve())})
    active = {"schema": SCHEMA, "release_id": registry["release_id"], "registry_sha256": registry_hash,
              "synced_at": utc_now(), "models": local_models}
    if previous and previous.get("registry_sha256") != registry_hash:
        atomic_json(cache / "history" / f"{previous['release_id']}_{uuid4().hex[:6]}.json", previous)
    atomic_json(active_path, active)
    return {"active": active, "changed": not previous or previous.get("registry_sha256") != registry_hash}


def rollback_cache(cache: Path) -> dict:
    histories = sorted((Path(cache) / "history").glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in histories:
        previous = read_json(path)
        if all(verify_files(Path(model["local_folder"]), model["files"]) for model in previous["models"]):
            atomic_json(Path(cache) / "active.json", previous)
            return previous
    raise ValueError("Нет целой предыдущей версии в локальном кеше")


def make_registry_model(source: Path, *, model_id: str, role: str, version: str, entrypoint: str, approval: dict,
                        base_model_id: str = "", base_version: str = "") -> dict:
    source = Path(source)
    identifier(model_id); identifier(version); relative(entrypoint)
    files = [{"path": p.relative_to(source).as_posix(), "size": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(source.rglob("*")) if p.is_file() and not p.is_symlink() and not p.name.startswith(".")]
    return {"id": model_id, "role": role, "version": version, "entrypoint": entrypoint, "files": files,
            "approval": approval, "base_model_id": base_model_id, "base_version": base_version}


def publish_model(hub: Path, source: Path, model: dict, *, baseline: bool = False):
    hub, source = Path(hub), Path(source)
    initialize_hub(hub)
    if not baseline:
        approval = model.get("approval", {})
        if approval.get("approved") is not True or not approval.get("independent_wells") or not approval.get("validation_report"):
            raise ValueError("Для новой рабочей версии нужен отчёт и независимые контрольные скважины")
        report = inside(hub, approval["validation_report"])
        if not report.is_file() or read_json(report).get("accepted") is not True:
            raise ValueError("Контрольный отчёт отсутствует или кандидат не принят")
    registry_file = hub / "registry/models.json"
    previous = read_json(registry_file) if registry_file.exists() else {"schema": SCHEMA, "models": []}
    registry = {**previous, "release_id": "r-" + uuid4().hex[:16], "published_at": utc_now(),
                "models": [item for item in previous["models"] if item["role"] != model["role"]] + [model]}
    validate_registry(registry)
    destination = hub / "models" / model["id"] / model["version"]
    if destination.exists():
        if not verify_files(destination, model["files"]):
            raise ValueError("Опубликованные версии не перезаписываются. Укажите новую версию")
    else:
        stage = destination.with_name(f".{destination.name}.{uuid4().hex}.publishing")
        stage.mkdir(parents=True)
        for item in model["files"]:
            origin, copied = inside(source, item["path"]), inside(stage, item["path"])
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, copied)
        if not verify_files(stage, model["files"]):
            raise ValueError("Файлы кандидата изменились во время публикации")
        stage.rename(destination)
    if registry_file.exists():
        atomic_json(hub / "registry/history" / (previous["release_id"] + ".json"), previous)
    atomic_json(registry_file, registry)


def submit_reviewed_packet(hub: Path, packet: Path) -> dict:
    hub, packet = Path(hub), Path(packet)
    if read_json(hub / "hub.json").get("schema") != "cv1-network-hub-v1":
        raise ValueError("Общее хранилище не подготовлено администратором")
    metadata = read_json(packet / "packet.json")
    if metadata.get("schema") != PACKET_SCHEMA or not metadata.get("annotations"):
        raise ValueError("В пакете нет проверенных интервалов")
    if any(row.get("geometry_approved") is not True for row in metadata["annotations"]):
        raise ValueError("Пакет содержит неподтверждённую геометрию")
    files = [{"path": path.relative_to(packet).as_posix(), "size": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(packet.rglob("*")) if path.is_file() and not path.is_symlink()]
    batch_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    destination = hub / "training/incoming" / batch_id
    if destination.exists():
        receipt = read_json(destination / "receipt.json")
        if receipt.get("files") != files or not verify_files(destination, files):
            raise ValueError("Повреждён ранее отправленный пакет")
        return {"batch_id": batch_id, "duplicate": True, "annotations": len(metadata["annotations"])}
    stage = destination.with_name(".upload-" + uuid4().hex)
    stage.mkdir(parents=True)
    for item in files:
        source, copied = inside(packet, item["path"]), inside(stage, item["path"])
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
    if not verify_files(stage, files):
        raise ValueError("Не удалось проверить переданные файлы")
    atomic_json(stage / "receipt.json", {"schema": PACKET_SCHEMA, "batch_id": batch_id, "files": files,
                "received_at": utc_now(), "central_approved": False})
    stage.rename(destination)
    return {"batch_id": batch_id, "duplicate": False, "annotations": len(metadata["annotations"])}


def approve_packet(hub: Path, batch_id: str) -> Path:
    hub = Path(hub)
    identifier(batch_id)
    source = hub / "training/incoming" / batch_id
    receipt = read_json(source / "receipt.json")
    if not verify_files(source, receipt["files"]):
        raise ValueError("Входящий пакет повреждён")
    metadata = read_json(source / "packet.json")
    if any(row.get("geometry_approved") is not True for row in metadata["annotations"]):
        raise ValueError("Неподтверждённые интервалы не допускаются в учебную базу")
    destination = hub / "training/approved" / batch_id
    if not destination.exists():
        stage = destination.with_name(".approving-" + uuid4().hex)
        shutil.copytree(source, stage)
        atomic_json(stage / "approval.json", {"batch_id": batch_id, "approved_at": utc_now(), "central_approved": True})
        stage.rename(destination)
    # An accepted full snapshot replaces older accepted snapshots of this
    # exact project on this computer. Historical packets are retained.
    if metadata.get("project_key"):
        for previous in (hub / "training/approved").iterdir():
            if previous == destination or not (previous / "approval.json").is_file():
                continue
            old = read_json(previous / "packet.json")
            if old.get("project_key") == metadata["project_key"] and old.get("computer") == metadata.get("computer"):
                atomic_json(previous / "approval.json", {"central_approved": False, "superseded_by": batch_id, "updated_at": utc_now()})
    return destination


def materialize_packet(snapshot: dict, outbox: Path) -> Path:
    """Freeze source photos and current expert confirmations before upload."""
    annotations, photos = [], {}
    for photo in snapshot["photos"]:
        source = Path(photo["source"])
        image_hash = digest(source)
        relative_image = "images/" + image_hash + source.suffix.casefold()
        photos[relative_image] = source
        for row in photo["annotations"]:
            if row.get("geometry_approved") is not True:
                continue
            annotations.append({**row, "image": relative_image, "source_photo_sha256": image_hash,
                                "canvas_size": photo["canvas_size"]})
    if not annotations:
        raise ValueError("Нет подтверждённых интервалов для передачи")
    packet = {"schema": PACKET_SCHEMA, "computer": snapshot["computer"], "project": snapshot["project"], "project_key": snapshot.get("project_key", ""),
              "annotations": annotations}
    packet_id = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    destination = Path(outbox) / packet_id
    if (destination / "packet.json").exists():
        return destination
    stage = destination.with_name(".packet-" + uuid4().hex)
    stage.mkdir(parents=True)
    for name, source in photos.items():
        target = inside(stage, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected = Path(name).stem
        if digest(target) != expected:
            raise ValueError("Фото изменилось во время подготовки пакета")
    atomic_json(stage / "packet.json", packet)
    stage.rename(destination)
    return destination
