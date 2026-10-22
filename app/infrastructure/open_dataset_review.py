"""Explicit specialist review of source-direct pairs and well-disjoint exports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _root(preparation: dict) -> Path:
    if not preparation.get("output_dir"):
        raise ValueError("В результате подготовки нет папки материалов")
    return Path(preparation["output_dir"]).resolve()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".review-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reviews(preparation: dict) -> dict:
    path = _root(preparation) / "expert_reviews.json"
    if not path.exists():
        return {"schema": "kern-open-dataset-expert-reviews-v1", "records": {}}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Файл правок специалиста не читается; сохранённые правки не перезаписаны") from exc
    if not isinstance(result, dict) or result.get("schema") != "kern-open-dataset-expert-reviews-v1" or not isinstance(result.get("records"), dict):
        raise ValueError("Неизвестный формат сохранённых правок специалиста")
    return result


def _candidate_signature(row: dict) -> str:
    source = {key: value for key, value in row.items() if key not in {"review", "image_path"}}
    return hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _original_preparation_roots(root: Path) -> list[Path]:
    """Read original roots as relocation hints without changing source reports."""
    roots = []
    for filename in ("report.json", "summary.json"):
        report = root / filename
        if not report.is_file():
            continue
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = payload.get("output_dir") if isinstance(payload, dict) else None
        if isinstance(value, str) and value and Path(value).is_absolute():
            roots.append(Path(value))
    return roots


def _candidate_image_path(root: Path, value, original_roots: list[Path]) -> Path:
    image = Path(str(value or ""))
    if not image.is_absolute():
        return (root / image).resolve()
    for previous in original_roots:
        try:
            relocated = (root / image.relative_to(previous)).resolve()
        except ValueError:
            continue
        # Prefer the copied library even if the former PC/path is still online.
        if relocated.is_relative_to(root) and relocated.is_file():
            return relocated
    return image.resolve()


def read_candidates(preparation: dict) -> list[dict]:
    """Read source rows plus separate drafts; a changed source revokes old approval."""
    root = _root(preparation)
    manifest = Path(preparation.get("manifest") or root / "candidates.jsonl")
    if not manifest.is_absolute():
        manifest = root / manifest
    reviews = _reviews(preparation)["records"]
    original_roots = _original_preparation_roots(root)
    rows, seen = [], set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise ValueError(f"Некорректная исходная запись: строка {number}")
        record_id = str(row["metadata"].get("record_id") or row.get("record_id") or "")
        if not record_id or record_id in seen:
            raise ValueError(f"Пустой или повторяющийся идентификатор: строка {number}")
        seen.add(record_id)
        image_path = _candidate_image_path(root, row.get("image"), original_roots)
        row = dict(row, record_id=record_id)
        signature = _candidate_signature(row)
        review = dict(reviews.get(record_id) or {})
        if review.get("candidate_signature") != signature:
            review["approved"] = False
            review["source_changed"] = bool(review.get("candidate_signature"))
        row.update(image_path=str(image_path), review=review)
        rows.append(row)
    return rows


def _image_hash(path: Path) -> str:
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert("RGB")
        digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def _finite(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Нет проверяемых числовых границ интервала") from exc
    if not math.isfinite(result):
        raise ValueError("Некорректные границы интервала")
    return result


def _validate(row: dict, answer_ru: str) -> str:
    if not answer_ru.strip() or not re.search(r"[А-Яа-яЁё]", answer_ru):
        raise ValueError("Введите собственное проверенное описание на русском языке")
    meta = row["metadata"]
    if meta.get("needs_interval_crop"):
        raise ValueError("На фото показан более длинный интервал. Сначала подготовьте вырезку нужного слоя и повторно добавьте пару")
    image = Path(row["image_path"])
    if not image.is_file():
        raise ValueError("Исходное фото не найдено")
    top, bottom = _finite(meta.get("depth_top_m")), _finite(meta.get("depth_bottom_m"))
    if bottom <= top:
        raise ValueError("Подошва должна быть глубже кровли")
    if meta.get("pairing_status") == "source_arrays_row_aligned":
        if not meta.get("pixel_alignment_verified"):
            raise ValueError("Не подтверждено соответствие строк фото и исходных глубин")
        start, end = _finite(meta.get("row_top")), _finite(meta.get("row_bottom"))
        crop_start, crop_end = _finite(meta.get("crop_row_top")), _finite(meta.get("crop_row_bottom"))
        if not start <= crop_start < crop_end <= end:
            raise ValueError("Вырезка выходит за исходный однородный интервал")
        if not math.isclose(top, _finite(meta.get("crop_depth_top_m")), abs_tol=1e-7) or not math.isclose(bottom, _finite(meta.get("crop_depth_bottom_m")), abs_tol=1e-7):
            raise ValueError("Глубины записи не соответствуют вырезке")
        run_top = _finite(meta.get("source_run_depth_top_m", top))
        run_bottom = _finite(meta.get("source_run_depth_bottom_m", bottom))
        if not run_top <= top < bottom <= run_bottom:
            raise ValueError("Глубины вырезки выходят за исходный интервал")
    elif meta.get("pairing_status") == "source_identifier_and_depth_matched_unreviewed":
        if not math.isclose(top, _finite(meta.get("photo_depth_top_m")), abs_tol=1e-7) or not math.isclose(bottom, _finite(meta.get("photo_depth_bottom_m")), abs_tol=1e-7):
            raise ValueError("Интервал описания не равен интервалу исходного фото")
    else:
        raise ValueError("Нет надёжного исходного сопоставления фото и интервала")
    try:
        return _image_hash(image)
    except (OSError, ValueError) as exc:
        raise ValueError("Изображение не читается") from exc


def save_review(preparation: dict, record_id: str, answer_ru: str, approved: bool) -> dict:
    """Save a draft or an explicit approval; source texts are never overwritten."""
    rows = read_candidates(preparation)
    row = next((item for item in rows if item["record_id"] == record_id), None)
    if row is None:
        raise ValueError("Исходная пара не найдена")
    answer_ru = str(answer_ru or "").strip()
    image_hash = _validate(row, answer_ru) if approved else ""
    review = {"answer_ru": answer_ru, "approved": bool(approved),
              "candidate_signature": _candidate_signature(row), "image_pixel_sha256": image_hash,
              "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "approval_origin": "explicit_local_specialist_review" if approved else "draft"}
    document = _reviews(preparation)
    document["records"][record_id] = review
    _atomic_json(_root(preparation) / "expert_reviews.json", document)
    return review


_SPLIT_SCHEMA = "kern-open-dataset-well-splits-v1"


def _well(value) -> str:
    return str(value or "").strip().casefold()


def _source_id(value) -> str:
    source = str(value or "").strip().casefold()
    return {"ukcs_q204_q205_coremdlr": "ukcs", "ukcs_q204_q205": "ukcs"}.get(source, source)


def _split_context(preparation: dict) -> tuple[Path, Path, list[Path], str]:
    """Share assignments across preparations of one source, retaining portability."""
    root = _root(preparation)
    source = _source_id(preparation.get("source_id"))
    if root.parent.name == "prepared":
        scope = root.parent.parent
        source = source or _source_id(scope.name)
        return scope / "well_split_assignments.json", scope, sorted(p for p in root.parent.iterdir() if p.is_dir()), source
    if root.parent.name == "open_dataset_review" and source:
        scope = root.parent
        legacy_roots = [root]
        for candidate in sorted(scope.iterdir()):
            if not candidate.is_dir() or candidate == root:
                continue
            report = candidate / "report.json"
            if report.is_file():
                try:
                    original = json.loads(report.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ValueError("Не читается паспорт соседней подготовки; принадлежность истории к источнику не проверена") from exc
                if isinstance(original, dict) and _source_id(original.get("source_id")) == source:
                    legacy_roots.append(candidate)
        safe_name = re.sub(r"[^a-z0-9_-]", "_", source)
        return scope / f"well_split_assignments.{safe_name}.json", scope, legacy_roots, source
    return root / "well_split_assignments.json", root, [root], source


def _well_assignments(preparation: dict, wells: list[str]) -> dict:
    """Keep earlier train membership out of every later holdout export.

    Older exports are evidence even if their original absolute paths no longer
    exist. Contradictory history needs investigation; choosing a new split would
    conceal possible exposure of a control well to a previous adapter.
    """
    path, scope, legacy_roots, source_id = _split_context(preparation)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Не читается история разделения скважин; назначения не перезаписаны") from exc
        if not isinstance(state, dict) or state.get("schema") != _SPLIT_SCHEMA or not isinstance(state.get("assignments"), dict):
            raise ValueError("Неизвестный формат истории разделения скважин")
        if state.get("source_id", "") != source_id:
            raise ValueError("История разделения относится к другому источнику")
        for well, assignment in state["assignments"].items():
            if not well or _well(well) != well or not isinstance(assignment, dict) or assignment.get("split") not in {"train", "holdout"}:
                raise ValueError("Повреждена история разделения скважин; назначения не перезаписаны")
        changed = False
    else:
        state = {"schema": _SPLIT_SCHEMA, "source_id": source_id, "created_at": now, "assignments": {},
                 "policy": "immutable_whole_well_membership; new_wells_sha256_threshold_20_percent"}
        assignments = state["assignments"]

        def remember(value, split: str, evidence: dict) -> None:
            well = _well(value)
            if not well:
                raise ValueError("В старом экспорте нет идентификатора скважины; разделение не изменено")
            previous = assignments.get(well)
            if previous and previous["split"] != split:
                raise ValueError(f"Скважина {well} встречалась и в обучении, и в контроле старых экспортов. Независимый контроль нельзя подтвердить; история сохранена")
            if previous is None:
                previous = assignments[well] = {"split": split, "assigned_at": now,
                                                "origin": "legacy_export_migration", "evidence": []}
            if evidence not in previous["evidence"]:
                previous["evidence"].append(evidence)

        directories = sorted(directory for legacy_root in legacy_roots
                             for directory in (legacy_root / "reviewed_exports").glob("*"))
        for directory in directories:
            if not directory.is_dir():
                continue
            report_path = directory / "export_report.json"
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ValueError("Не читается старый паспорт экспорта; история разделения не изменена") from exc
                if not isinstance(report, dict):
                    raise ValueError("Некорректный старый паспорт экспорта")
                evidence = {"path": report_path.relative_to(scope).as_posix(),
                            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}
                for split, key in (("train", "train_wells"), ("holdout", "holdout_wells")):
                    values = report.get(key, [])
                    if not isinstance(values, list):
                        raise ValueError("Некорректный список скважин старого экспорта")
                    for well in values:
                        remember(well, split, evidence)
            # Use local files instead of possibly obsolete paths in the report.
            # Even an interrupted export's train manifest is prior exposure.
            for split, name in (("train", "train.jsonl"), ("holdout", "holdout.jsonl")):
                manifest = directory / name
                if not manifest.is_file():
                    continue
                evidence = {"path": manifest.relative_to(scope).as_posix(),
                            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
                try:
                    for line in manifest.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            row = json.loads(line)
                            remember(row.get("metadata", {}).get("well_id"), split, evidence)
                except (OSError, AttributeError, TypeError, ValueError) as exc:
                    raise ValueError("Не удалось восстановить разделение из старого экспорта: " + str(exc)) from exc
        changed = True
    assignments = state["assignments"]
    new_wells = sorted(set(wells) - set(assignments), key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    if new_wells:
        known_splits = {assignments[well]["split"] for well in wells if well in assignments}
        reserved = {}
        first_cohort = not assignments
        if first_cohort:
            # Bootstrap a useful first cohort; these choices are persisted once.
            size = max(1, round(len(new_wells) * 0.2))
            reserved = {well: "holdout" if index < size else "train" for index, well in enumerate(new_wells)}
        elif not known_splits:
            reserved[new_wells[0]], reserved[new_wells[-1]] = "holdout", "train"
        elif "holdout" not in known_splits:
            reserved[new_wells[0]] = "holdout"
        elif "train" not in known_splits:
            reserved[new_wells[-1]] = "train"
        for well in new_wells:
            fraction = int(hashlib.sha256(well.encode("utf-8")).hexdigest(), 16) / 2**256
            assignments[well] = {"split": reserved.get(well, "holdout" if fraction < .2 else "train"),
                                 "assigned_at": now,
                                 "origin": "first_export_hash_order" if first_cohort
                                           else "new_well_missing_split" if well in reserved else "new_well_sha256_threshold"}
        changed = True
    if changed:
        state["updated_at"] = now
        _atomic_json(path, state)
    active = {well: assignments[well] for well in wells}
    if {value["split"] for value in active.values()} != {"train", "holdout"}:
        raise ValueError("Среди подтверждённых пар нет обеих ранее закреплённых частей: обучения и контроля. Подтвердите скважины недостающей части или добавьте новую; существующие назначения сохранены")
    return active


def export_reviewed_pairs(preparation: dict) -> dict:
    """Export only unchanged explicit approvals, with whole-well holdout isolation."""
    approved, skipped = [], []
    image_groups = {}
    duplicate_pairs = {}
    for row in read_candidates(preparation):
        review = row["review"]
        if not review.get("approved"):
            continue
        image_hash = _validate(row, review.get("answer_ru", ""))
        if image_hash != review.get("image_pixel_sha256"):
            raise ValueError("После проверки изменилось фото. Повторно подтвердите пару")
        well = str(row["metadata"].get("well_id") or "").strip().casefold()
        if not well:
            raise ValueError("В проверенной паре нет идентификатора скважины")
        if image_hash in image_groups and image_groups[image_hash] != well:
            raise ValueError("Одинаковое изображение указано в разных скважинах. Исправьте принадлежность перед разделением обучения и контроля")
        image_groups[image_hash] = well
        duplicate_key = (well, image_hash)
        if duplicate_key in duplicate_pairs:
            previous = duplicate_pairs[duplicate_key]
            if previous != review["answer_ru"]:
                raise ValueError("Одному изображению назначены разные проверенные описания")
            skipped.append(row["record_id"])
            continue
        duplicate_pairs[duplicate_key] = review["answer_ru"]
        approved.append((row, well, image_hash))
    wells = sorted({well for _, well, _ in approved})
    if len(wells) < 2:
        raise ValueError("Для независимого контроля подтвердите пары минимум из двух разных скважин. Черновики и подтверждения сохранены")
    assignments = _well_assignments(preparation, wells)
    holdout_wells = {well for well, value in assignments.items() if value["split"] == "holdout"}
    training, holdout = [], []
    revision_rows = []
    for row, well, image_hash in approved:
        review = row["review"]
        is_holdout = well in holdout_wells
        metadata = dict(row["metadata"])
        metadata.update(training_approved=not is_holdout, geology_approved=True,
                        review_status="specialist_approved", label_provenance="source_direct_specialist_corrected_russian_answer",
                        record_type="image_text", source_group=f"{metadata.get('source_id', '')}:{well}",
                        well_id=well, split="holdout" if is_holdout else "train",
                        split_assignment_provenance=dict(assignments[well]),
                        reviewed_source_photo=row.get("image", ""),
                        image_pixel_sha256=image_hash, reviewed_at=review["reviewed_at"],
                        approval_origin=review["approval_origin"], original_source_label=row.get("source_label", ""),
                        original_source_description=row.get("original_description", ""))
        result = {"image": row["image_path"], "conversations": [
            {"from": "human", "value": "<image>\nОпиши видимые признаки породы на русском языке. Не выдумывай невидимые детали."},
            {"from": "gpt", "value": review["answer_ru"]}], "metadata": metadata}
        (holdout if is_holdout else training).append(result)
        revision_rows.append([row["record_id"], image_hash, review["answer_ru"], metadata["split"],
                              review["candidate_signature"], review["reviewed_at"]])
    train_hashes = {r["metadata"]["image_pixel_sha256"] for r in training}
    if train_hashes & {r["metadata"]["image_pixel_sha256"] for r in holdout}:
        raise ValueError("Фото пересекаются между обучением и контролем")
    revision = hashlib.sha256(json.dumps({"schema": "persistent-well-split-v2-relative-images", "rows": sorted(revision_rows)}, ensure_ascii=False).encode("utf-8")).hexdigest()
    directory = _root(preparation) / "reviewed_exports" / revision[:20]
    directory.mkdir(parents=True, exist_ok=True)
    images_directory = directory / "images"
    images_directory.mkdir(exist_ok=True)
    for row in training + holdout:
        source_image = Path(row["image"])
        image_hash = row["metadata"]["image_pixel_sha256"]
        snapshot = images_directory / f"{image_hash}{source_image.suffix.casefold()}"
        if not snapshot.exists():
            shutil.copyfile(source_image, snapshot)
        if _image_hash(snapshot) != image_hash:
            raise ValueError("Сохранённое фото экспорта не соответствует проверенной паре")
        row["image"] = snapshot.relative_to(directory).as_posix()
    for name, rows in (("train.jsonl", training), ("holdout.jsonl", holdout)):
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path = directory / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Существующий экспорт отличается; сохранённый набор не перезаписан")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    report = {"schema": "kern-open-reviewed-photo-pairs-v1", "manifest": str(directory / "train.jsonl"),
              "holdout_manifest": str(directory / "holdout.jsonl"), "count": len(training),
              "counts": {"train": len(training), "holdout": len(holdout), "wells": len(wells),
                         "duplicates_removed": len(skipped)}, "train_wells": sorted(set(wells) - holdout_wells),
              "holdout_wells": sorted(holdout_wells), "source_digest": revision,
              "split_strategy": "whole_well", "image_hash_overlap": 0,
              "split_assignment_policy": "persistent_whole_well_v1",
              "split_assignments_file": str(_split_context(preparation)[0]),
              "split_assignment_provenance": assignments,
              "images_snapshotted": True,
              "limitation": "Контроль выделен по скважинам; это не измерение точности и не гарантия представительности всех фаций."}
    _atomic_json(directory / "export_report.json", report)
    return report
