"""Explicit geologist corrections for local chat memory and Qwen training."""

from __future__ import annotations

import hashlib
import json
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
    if not previous or previous == current:
        return
    rows = read_corrections()
    changed = False
    for row in rows:
        if row.get("scope") == "project" and row.get("project") == previous:
            row["project"] = current
            changed = True
    if changed:
        path = _root() / "corrections.jsonl"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
        )
        temporary.replace(path)


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
    return row


def exact_correction(question: str, *, operation: str = "chat", image_path: str = "", project: str = "", well: str = "") -> str | None:
    key = _normalized(question)
    matches = [
        row for row in reversed(read_corrections())
        if row.get("operation") == operation
        and _normalized(str(row.get("question") or "")) == key
        and (row.get("scope") == "global" or str(row.get("project") or "") == project)
        and (operation != "chat" or row.get("scope") == "global" or str(row.get("well") or "") == well)
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
        if row.get("scope") != "global" and str(row.get("project") or "") != project:
            continue
        if row.get("scope") != "global" and str(row.get("well") or "") != well:
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
