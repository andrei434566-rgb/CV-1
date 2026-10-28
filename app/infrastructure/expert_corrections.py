"""Explicit geologist corrections for local chat memory and Qwen training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.runtime_paths import user_data_root


def _root() -> Path:
    return user_data_root() / "vlm_training" / "expert_feedback"


def _normalized(question: str) -> str:
    return " ".join(re.findall(r"[\wа-яё]+", question.casefold(), flags=re.UNICODE))


def _image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_corrections() -> list[dict]:
    path = _root() / "corrections.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("approved") is True:
                rows.append(row)
    return rows


def reassign_project(previous: str, current: str) -> None:
    """Keep local project-scoped memory when an unsaved project is first saved."""
    if not previous or not current or previous == current:
        return
    # Do not discard malformed/unapproved rows when changing a project key.
    # Earlier versions rewrote only read_corrections(), losing those records.
    for name in ("corrections.jsonl", "interval_edits.jsonl"):
        path = _root() / name
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        changed = False
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("scope") == "project" and row.get("project") == previous:
                row["project"] = current
                lines[index] = json.dumps(row, ensure_ascii=False) + "\n"
                changed = True
        if changed:
            temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
            try:
                temporary.write_text("".join(lines), encoding="utf-8")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def save_correction(
    question: str,
    model_answer: str,
    corrected_answer: str,
    *,
    operation: str = "chat",
    image_path: str = "",
    project: str = "",
    well: str = "",
    general: bool = False,
) -> dict:
    question, model_answer, corrected_answer = (
        str(value or "").strip() for value in (question, model_answer, corrected_answer)
    )
    if not question or not corrected_answer:
        raise ValueError("Для обучения нужны вопрос и исправленный ответ специалиста.")
    if corrected_answer == model_answer:
        raise ValueError("Исправленный ответ совпадает с исходным; внесите поправку перед сохранением.")
    if operation not in {"chat", "photo_chat"}:
        raise ValueError("Поправки сохраняются для текстового чата или ответа по фото.")
    if general and operation == "photo_chat":
        raise ValueError("Поправка по фото не может быть общей для других проектов и скважин.")
    if not general and (not str(project).strip() or not str(well).strip()):
        raise ValueError("Для поправки укажите проект и скважину.")
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    source_image = ""
    image_digest = ""
    saved_image = ""
    if operation == "photo_chat":
        source = Path(image_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError("Исходное фото недоступно. Прикрепите его снова перед сохранением поправки.")
        images = root / "images"
        images.mkdir(exist_ok=True)
        source_image = str(source.resolve())
        image_digest = _image_hash(source)
        target = images / f"{identifier}{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, target)
        saved_image = f"images/{target.name}"
    row = {
        "id": identifier,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "question": question,
        "model_answer": model_answer,
        "corrected_answer": corrected_answer,
        "source_image": source_image,
        "image_sha256": image_digest,
        "saved_image": saved_image,
        "project": project,
        "well": well,
        "scope": "global" if general else "project",
        "approved": True,
    }
    with (root / "corrections.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return row


def exact_correction(question: str, *, operation: str = "chat", image_path: str = "", project: str = "", well: str = "") -> str | None:
    key = _normalized(question)
    matches = [
        row for row in reversed(read_corrections())
        if row.get("operation") == operation
        and _normalized(str(row.get("question") or "")) == key
        and ((operation == "chat" and row.get("scope") == "global") or (
            bool(project and well) and row.get("scope") == "project"
            and str(row.get("project") or "") == project
            and str(row.get("well") or "") == well
        ))
    ]
    if not matches:
        return None
    image_digest = ""
    if operation == "photo_chat":
        source = Path(image_path)
        if not source.is_file():
            return None
        image_digest = _image_hash(source)
    for row in matches:
        if operation == "photo_chat" and image_digest != row.get("image_sha256"):
            continue
        return str(row.get("corrected_answer") or "") or None
    return None


def related_text_corrections(question: str, *, project: str = "", well: str = "", limit: int = 3) -> list[dict[str, str]]:
    """Show only close text examples to Qwen; photo labels never leak across images."""
    words = {word for word in _normalized(question).split() if len(word) >= 4}
    if not words:
        return []
    scored = []
    seen: set[tuple[str, str, str]] = set()
    for row in reversed(read_corrections()):
        if row.get("operation") != "chat":
            continue
        if row.get("scope") not in {"project", "global"}:
            continue
        if row.get("scope") != "global" and (not project or str(row.get("project") or "") != project):
            continue
        if row.get("scope") != "global" and (not well or str(row.get("well") or "") != well):
            continue
        key = (str(row.get("scope") or "project"), str(row.get("project") or "") + "|" + str(row.get("well") or ""), _normalized(str(row.get("question") or "")))
        if key in seen:
            continue
        seen.add(key)
        known = {word for word in _normalized(str(row.get("question") or "")).split() if len(word) >= 4}
        if not known:
            continue
        similarity = len(words & known) / len(words | known)
        if similarity >= 0.60:
            scored.append((similarity, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"question": str(row["question"])[:400], "answer": str(row["corrected_answer"])[:1000]}
        for _, row in scored[:max(0, limit)]
    ]


def correction_counts() -> dict[str, int]:
    rows = read_corrections()
    return {
        "text": sum(row.get("operation") == "chat" for row in rows),
        "text_global": sum(row.get("operation") == "chat" and row.get("scope") == "global" for row in rows),
        "photo": sum(row.get("operation") == "photo_chat" for row in rows),
    }


def _interval_identity(record, detection, *, project: str, well: str) -> dict[str, str]:
    """Bind an edit to one project, well, original photo and interval geometry."""
    project, well = str(project or "").strip(), str(well or "").strip()
    if not project or not well or str(record.well_name or "").strip() != well:
        raise ValueError("Для памяти интервала нужны совпадающие проект и скважина.")
    path = Path(record.path)
    if not path.is_file():
        raise FileNotFoundError("Исходное фото интервала недоступно.")
    width, height = record.pixmap.width(), record.pixmap.height()
    if width <= 0 or height <= 0 or len(detection.polygon) < 3:
        raise ValueError("Для памяти интервала нужен корректный контур на исходном фото.")
    geometry = {
        "polygon_relative": [[round(float(p.x()) / width, 6), round(float(p.y()) / height, 6)]
                             for p in detection.polygon],
        "depth_from": detection.depth_from,
        "depth_to": detection.depth_to,
    }
    data = json.dumps(geometry, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"project": project, "well": well, "image_sha256": _image_hash(path),
            "interval_sha256": hashlib.sha256(data.encode("utf-8")).hexdigest()}


def expert_memory_digest(detection) -> str:
    """Bind a memory confirmation to the exact text, facies, depth and geometry.

    This stamp is deliberately separate from the stricter training approval.
    It does not change which examples are exportable for model training.
    """
    from app.infrastructure.vlm_training_data import expert_description_digest
    return expert_description_digest(detection)


def expert_memory_is_confirmed(detection) -> bool:
    """Whether a specialist confirmed this unchanged description for memory."""
    attrs = detection.attributes or {}
    if not (attrs.get("__expert_memory_confirmed") == "1"
            and attrs.get("__expert_memory_sha256")
            and any(str(attrs.get(key) or "").strip() for key in (
                "__vlm_description", "Краткое описание", "Литологическое описание (16 параметров)"))):
        return False
    try:
        return attrs["__expert_memory_sha256"] == expert_memory_digest(detection)
    except (TypeError, ValueError):
        return False


def save_interval_edit(record, detection, *, project: str, well: str) -> dict:
    """Append a specialist's explicit edit, preserving its review state.

    The UI must call this only when the specialist saves a manual edit, not
    when applying model output. Unverified edits are persisted for the same
    interval but never promoted to approved training data by this function.
    """
    from app.infrastructure.vlm_training_data import expert_description_is_confirmed

    identity = _interval_identity(record, detection, project=project, well=well)
    attributes = {str(k): str(v) for k, v in (detection.attributes or {}).items()}
    row = {
        "schema": "kern-expert-interval-edit-v1", "id": uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "project", **identity, "photo_name": Path(record.path).name,
        "facies": str(detection.label), "attributes": attributes,
        "depth_from": detection.depth_from, "depth_to": detection.depth_to,
        "training_ready": bool(detection.training_ready),
        # An explicit memory uncheck takes precedence over an older training
        # confirmation; older projects without the new stamp retain compatibility.
        "description_expert_confirmed": bool(
            expert_memory_is_confirmed(detection) if "__expert_memory_confirmed" in attributes
            else expert_description_is_confirmed(detection)
        ),
        "source": "explicit_manual_interval_edit",
    }
    # Validate before opening the append target, so failed serialization cannot
    # create an incomplete training record.
    line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "interval_edits.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return row


def exact_interval_edit(record, detection, *, project: str, well: str, require_confirmed: bool = True) -> dict | None:
    """Return only the latest explicit edit for this exact project/well/photo/interval.

    By default only a separately confirmed description can be supplied as
    context for the assistant. Editing a different interval or an unreviewed
    draft must never be silently recycled as an expert answer.
    """
    path = _root() / "interval_edits.jsonl"
    if not path.is_file():
        return None
    try:
        identity = _interval_identity(record, detection, project=project, well=well)
    except (OSError, TypeError, ValueError):
        return None
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("schema") == "kern-expert-interval-edit-v1" and all(
                row.get(key) == value for key, value in identity.items()
            ):
                rows.append(row)
    if not rows:
        return None
    # A newer unchecked edit revokes an earlier approval; never fall back to
    # a stale approved version when the specialist changes the text.
    latest = rows[-1]
    return latest if not require_confirmed or latest.get("description_expert_confirmed") is True else None


def export_correction_manifest(kind: str, destination: Path) -> dict:
    """Snapshot only explicit expert edits in the existing Qwen JSONL format."""
    if kind not in {"text", "photo"}:
        raise ValueError("Выберите текстовые или фото-поправки.")
    operation = "chat" if kind == "text" else "photo_chat"
    latest: dict[tuple[str, str, str, str], dict] = {}
    for row in read_corrections():
        key = (
            str(row.get("scope") or "project"),
            (str(row.get("project") or "") + "|" + str(row.get("well") or "")) if row.get("scope") != "global" else "",
            _normalized(str(row.get("question") or "")),
            str(row.get("image_sha256") or ""),
        )
        latest[key] = row
    rows = []
    for correction in latest.values():
        if correction.get("operation") != operation:
            continue
        if kind == "text" and correction.get("scope") != "global":
            continue
        image = str(correction.get("saved_image") or "")
        image_path = _root() / image if image and not Path(image).is_absolute() else Path(image)
        if kind == "photo" and not image_path.is_file():
            continue
        record = {
            "conversations": [
                {"from": "human", "value": ("<image>" if image else "") + str(correction["question"])},
                {"from": "gpt", "value": str(correction["corrected_answer"])},
            ],
            "metadata": {
                "training_approved": True,
                "geology_approved": True,
                "record_type": "image_text" if image else "text_only",
                "label_provenance": "explicit_geologist_chat_correction",
                "source_group": "expert_feedback",
                "source_well": correction.get("well") or "",
                "correction_id": correction["id"],
                "project": correction.get("project") or "",
            },
        }
        if image:
            record["image"] = str(image_path.resolve())
        rows.append(record)
    if not rows:
        raise ValueError("Пока нет сохранённых поправок специалиста этого типа.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    return {"manifest": destination, "count": len(rows)}
