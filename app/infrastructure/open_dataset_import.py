"""Prepare source-specific open core datasets without Qt or network access.

``prepare_dataset(source_id, source_paths, output_dir, progress=None, cancel=None)``
reads ZIPs or extracted directories in place. ``progress`` receives dictionaries
with stage/completed/total/message; ``cancel`` is a callable or threading.Event.
The destination must not exist. A sibling staging directory is atomically renamed
only after successful validation. No archive is executed or extracted wholesale.

The source class vocabularies are deliberately independent of the application's
facies catalogue. Classification records are labels, never invented descriptions.
UniBo's canonical targets are indexed semantic PNG masks (0 background, 1..6
facies, 255 ignored). YOLO polygons are an additional, potentially less detailed
export; tiny components omitted from that export are counted in the manifest.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_MEMBER_BYTES = 256 * 1024**2
MAX_ARCHIVE_BYTES = 32 * 1024**3
MAX_ENTRIES = 200_000
MAX_IMAGE_PIXELS = 48_000_000
TILE_SIZE = 768
MAX_SEGMENTATION_SIDE = 3072  # at most 16 tiles/photo, regardless of input size
PALETTE_TOLERANCE = 24.0
PALETTE_MARGIN = 4.0
IGNORE_INDEX = 255

UNIBO_CLASSES = [
    {"id": 0, "name": "Well-drained floodplain", "display_ru": "Хорошо дренируемая пойма", "abbreviation": "WDF", "rgb": [242, 229, 136], "mask_id": 1},
    {"id": 1, "name": "Poorly-drained floodplain", "display_ru": "Плохо дренируемая пойма", "abbreviation": "PDF", "rgb": [128, 193, 123], "mask_id": 2},
    {"id": 2, "name": "Prodelta", "display_ru": "Продельта", "abbreviation": "P", "rgb": [106, 69, 149], "mask_id": 3},
    {"id": 3, "name": "Peat layer", "display_ru": "Прослой торфа", "abbreviation": "PL", "rgb": [32, 75, 155], "mask_id": 4},
    {"id": 4, "name": "Fluvial sand", "display_ru": "Русловой песок", "abbreviation": "FS", "rgb": [242, 137, 21], "mask_id": 5},
    {"id": 5, "name": "Swamp", "display_ru": "Болото", "abbreviation": "Sw", "rgb": [102, 179, 45], "mask_id": 6},
]

SOURCE_INFO = {
    "unibo": {"title": "Sedimentary Facies Analysis and Segmentation", "url": "https://amsacta.unibo.it/id/eprint/7308/", "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/", "attribution": "Andrea Di Martino and Alessandro Amorosi, University of Bologna (2023), DOI 10.6092/unibo/amsacta/7308"},
    "corebreakout": {"title": "CoreBreakout BGS core-column annotations", "url": "https://github.com/rgmyr/corebreakout/releases/tag/v0.2", "license": "MIT (code/annotations); Open Government Licence (BGS core photographs)", "license_url": "https://www.bgs.ac.uk/information-hub/photos-and-images/", "attribution": "Ross Meyer / CoreBreakout; Contains British Geological Survey materials © UKRI. Preserve the original photograph year and source identifier when available."},
    "dcid": {"title": "Drill Core Image Dataset (DCID)", "url": "https://huggingface.co/datasets/168sir/drill-core-image-dataset", "license": "CC BY-NC 4.0", "license_url": "https://creativecommons.org/licenses/by-nc/4.0/", "attribution": "Jia-Yu Li, Ji-Zhou Tang et al. (2025), DOI 10.1016/j.petsci.2025.04.013"},
    "mendeley": {"title": "Lithology dataset", "url": "https://data.mendeley.com/datasets/b6rj29mrss/1", "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "attribution": "Fatimah Alzubaidi (2021), DOI 10.17632/b6rj29mrss.1"},
}


class DatasetPreparationCancelled(RuntimeError):
    """Preparation was cancelled; no complete destination was published."""


def _check_cancel(cancel) -> None:
    if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
        raise DatasetPreparationCancelled("Подготовка набора отменена.")


def _notify(progress, cancel, stage: str, completed: int, total: int, message: str) -> None:
    _check_cancel(cancel)
    if progress:
        progress({"stage": stage, "completed": completed, "total": total, "message": message})
    _check_cancel(cancel)


def _safe_member(name: str) -> PurePosixPath:
    name = name.replace("\\", "/")
    path = PurePosixPath(name)
    if not name or "\x00" in name or path.is_absolute() or any(p in {"..", ""} or ":" in p for p in path.parts):
        raise ValueError(f"Небезопасный путь в архиве: {name!r}")
    return path


def _slug(name: str) -> str:
    text = re.sub(r"[^\w.()-]+", "_", name, flags=re.UNICODE).strip(" ._")[:32]
    return (text or "unnamed") + "_" + hashlib.sha256(name.encode()).hexdigest()[:8]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _is_image(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return path.suffix.casefold() in IMAGE_SUFFIXES and not any(p == "__MACOSX" or p.startswith("._") for p in path.parts)


def _file_hash(path: Path, cancel=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            _check_cancel(cancel)
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _Entry:
    name: str
    source: Path
    size: int
    archive: zipfile.ZipFile | None = None
    disk: Path | None = None

    def read(self, cancel=None) -> bytes:
        if self.size > MAX_MEMBER_BYTES:
            raise ValueError(f"Файл превышает ограничение {MAX_MEMBER_BYTES} байт: {self.name}")
        chunks, total = [], 0
        with (self.archive.open(self.name) if self.archive else self.disk.open("rb")) as stream:
            while True:
                _check_cancel(cancel)
                chunk = stream.read(1024**2)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MEMBER_BYTES or total > self.size:
                    raise ValueError(f"Файл превышает заявленный размер: {self.name}")
                chunks.append(chunk)
        if total != self.size:
            raise ValueError(f"Неполный файл: {self.name}")
        return b"".join(chunks)

    def provenance(self) -> dict:
        return {"source_path": str(self.source), "member": self.name, "bytes": self.size}


class _Inputs:
    def __init__(self, paths: list[Path], cancel=None):
        self.paths, self.cancel = paths, cancel
        self.stack = ExitStack()
        self.entries: list[_Entry] = []
        self.sources: list[dict] = []

    def __enter__(self):
        try:
            for path in self.paths:
                _check_cancel(self.cancel)
                if path.is_symlink():
                    raise ValueError(f"Символические ссылки не поддерживаются: {path}")
                if path.is_dir():
                    files = []
                    for directory, dirs, names in os.walk(path, followlinks=False):
                        for name in dirs + names:
                            if (Path(directory) / name).is_symlink():
                                raise ValueError(f"Символическая ссылка в исходной папке: {name}")
                        for name in names:
                            disk = Path(directory) / name
                            if disk.suffix.casefold() in IMAGE_SUFFIXES | {".json", ".csv", ".rtf", ".md", ".txt"}:
                                files.append(disk)
                        if len(files) > MAX_ENTRIES:
                            raise ValueError("Слишком много файлов в исходной папке.")
                    self.entries.extend(_Entry(p.relative_to(path).as_posix(), path, p.stat().st_size, disk=p) for p in sorted(files))
                    self.sources.append({"path": str(path), "kind": "directory", "selected_files": len(files)})
                elif path.suffix.casefold() == ".zip":
                    archive = self.stack.enter_context(zipfile.ZipFile(path))
                    infos = archive.infolist()
                    if len(infos) > MAX_ENTRIES or sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES:
                        raise ValueError("Архив превышает ограничение числа файлов или распакованного объёма.")
                    names = set()
                    for info in infos:
                        _check_cancel(self.cancel)
                        safe = _safe_member(info.filename)
                        if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                            raise ValueError(f"Зашифрованный файл или ссылка в архиве: {info.filename}")
                        if safe.as_posix().casefold() in names:
                            raise ValueError(f"Повторяющийся путь в архиве: {info.filename}")
                        names.add(safe.as_posix().casefold())
                        if not info.is_dir():
                            self.entries.append(_Entry(info.filename, path, info.file_size, archive=archive))
                    self.sources.append({"path": str(path), "kind": "zip", "bytes": path.stat().st_size,
                                         "sha256": _file_hash(path, self.cancel), "members": len(infos)})
                elif path.is_file() and path.suffix.casefold() == ".csv":
                    self.entries.append(_Entry(path.name, path, path.stat().st_size, disk=path))
                    self.sources.append({"path": str(path), "kind": "well_group_mapping"})
                else:
                    raise ValueError(f"Выберите ZIP или папку с данными: {path}")
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def _decode(data: bytes, name: str) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        if image.width < 2 or image.height < 2 or image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Недопустимый размер изображения: {name}")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Многостраничное изображение не поддерживается: {name}")
        image.load()
        return image.convert("RGB")


def _pixel_hash(image: Image.Image) -> str:
    digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _original_split(name: str) -> str:
    aliases = {"train": "train", "training": "train", "training_subset": "train", "val": "val", "valid": "val", "validation": "val", "validation_subset": "val", "test": "test", "testing": "test", "test_subset": "test"}
    for part in PurePosixPath(name.replace("\\", "/")).parts[:-1]:
        if part.casefold() in aliases:
            return aliases[part.casefold()]
    return "unspecified"


def _group_splits(groups: set[str]) -> dict[str, str]:
    ordered = sorted(groups, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    if len(ordered) < 2:
        return {g: "train" for g in ordered}
    n_test = max(1, round(len(ordered) * .15)) if len(ordered) >= 3 else 0
    n_val = max(1, round(len(ordered) * .15))
    return {g: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, g in enumerate(ordered)}


def _save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=94, subsampling=0)


def _resize_pair(image: Image.Image, mask: np.ndarray | None = None):
    factor = min(1.0, MAX_SEGMENTATION_SIDE / max(image.size))
    size = max(2, round(image.width * factor)), max(2, round(image.height * factor))
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
        if mask is not None:
            mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)).copy()
    return image, mask, factor


def _palette_mask(image: Image.Image, cancel=None) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    palette = np.array([[0, 0, 0]] + [c["rgb"] for c in UNIBO_CLASSES], dtype=np.int32)
    target = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    codes = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    for index, (red, green, blue) in enumerate(palette):
        _check_cancel(cancel)
        target[codes == (int(red) << 16) | (int(green) << 8) | int(blue)] = index
    unknown = np.flatnonzero(target.ravel() == IGNORE_INDEX)
    flat_rgb, flat_target = rgb.reshape(-1, 3), target.ravel()
    for offset in range(0, len(unknown), 128 * 1024):
        _check_cancel(cancel)
        indices = unknown[offset:offset + 128 * 1024]
        colors = flat_rgb[indices].astype(np.int32)
        distances = ((colors[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2, dtype=np.int32)
        nearest = distances.argmin(axis=1)
        first_two = np.partition(distances, 1, axis=1)[:, :2]
        closest, second = np.sqrt(first_two.min(axis=1)), np.sqrt(first_two.max(axis=1))
        valid = (closest <= PALETTE_TOLERANCE) & ((second - closest) >= PALETTE_MARGIN)
        flat_target[indices[valid]] = nearest[valid].astype(np.uint8)
    return target


def _polygons(mask: np.ndarray, cancel=None) -> tuple[list[str], int]:
    """Export simple contours; split regions containing holes instead of filling them."""
    height, width = mask.shape
    lines, represented = [], 0
    for mask_id in range(1, 7):
        pending = [(mask == mask_id, 0, 0)]
        while pending:
            _check_cancel(cancel)
            binary, ox, oy = pending.pop()
            contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            holes = [i for i, h in enumerate(hierarchy[0]) if h[3] >= 0]
            if holes and min(binary.shape) > 4:
                x, y, w, h = cv2.boundingRect(contours[holes[0]])
                if binary.shape[1] >= binary.shape[0]:
                    cut = min(binary.shape[1] - 1, max(1, x + w // 2))
                    pending += [(binary[:, :cut], ox, oy), (binary[:, cut:], ox + cut, oy)]
                else:
                    cut = min(binary.shape[0] - 1, max(1, y + h // 2))
                    pending += [(binary[:cut], ox, oy), (binary[cut:], ox, oy + cut)]
                continue
            for contour, h in zip(contours, hierarchy[0]):
                if h[3] >= 0 or h[2] >= 0 or len(contour) < 3 or cv2.contourArea(contour) < 16:
                    continue
                raster = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(raster, [contour], -1, 1, -1)
                if np.any(raster.astype(bool) & ~binary):
                    continue
                represented += int(raster.sum())
                points = contour[:, 0, :].astype(float) + [ox, oy]
                coords = " ".join(f"{v:.8f}" for point in points for v in (point[0] / width, point[1] / height))
                lines.append(f"{mask_id - 1} {coords}")
    return lines, max(0, int(((mask > 0) & (mask < 7)).sum()) - represented)


def _prepare_unibo(inputs, stage, progress, cancel):
    images, masks = {}, {}
    for entry in inputs.entries:
        parts = [p.casefold() for p in PurePosixPath(entry.name.replace("\\", "/")).parts]
        if not _is_image(entry.name):
            continue
        bucket = images if "images" in parts[:-1] else masks if "annotations" in parts[:-1] else None
        if bucket is not None:
            key = PurePosixPath(entry.name.replace("\\", "/")).name.casefold()
            if key in bucket:
                raise ValueError(f"Неоднозначное имя изображения или маски UniBo: {key}")
            bucket[key] = entry
    if not images or set(images) != set(masks):
        missing = sorted(set(images) ^ set(masks))[:8]
        raise ValueError(f"Нужны все соответствующие Images/Annotations UniBo. Несовпадения: {missing}")
    wells = {}
    for name in images:
        match = re.match(r"^(S\d+)(?:_|\.)", name, flags=re.I)
        if not match:
            raise ValueError(f"Не удалось определить скважину UniBo по имени: {name}")
        wells[name] = match.group(1).upper()
    assignments = _group_splits(set(wells.values()))
    rows, issues, seen, class_counts = [], [], {}, Counter()
    original_groups = defaultdict(set)
    for index, name in enumerate(sorted(images), 1):
        _notify(progress, cancel, "preparing", index - 1, len(images), name)
        entry, ann = images[name], masks[name]
        data, ann_data = entry.read(cancel), ann.read(cancel)
        image, annotation = _decode(data, name), _decode(ann_data, ann.name)
        if image.size != annotation.size:
            issues.append({"reason": "image_annotation_size_mismatch_quarantined", "image_size": list(image.size),
                           "annotation_size": list(annotation.size), "source": entry.provenance(), "annotation": ann.provenance()})
            continue
        digest, annotation_digest = _pixel_hash(image), hashlib.sha256(ann_data).hexdigest()
        if digest in seen:
            previous = seen[digest]
            if previous != (annotation_digest, wells[name]):
                raise ValueError(f"Копия фото UniBo с другой маской или скважиной: {name}")
            issues.append({"reason": "duplicate_photo", **entry.provenance()})
            continue
        seen[digest] = annotation_digest, wells[name]
        split, original_split = assignments[wells[name]], _original_split(entry.name)
        original_groups[wells[name]].add(original_split)
        raw_ann = stage / "original_annotations" / f"{digest}.png"
        raw_ann.parent.mkdir(parents=True, exist_ok=True)
        raw_ann.write_bytes(ann_data)
        width, height = image.size
        image, _, factor = _resize_pair(image)
        # Mapping is pointwise, so nearest-neighbour resize commutes with it.
        # Avoid classifying millions of source pixels that would be discarded.
        if annotation.size != image.size:
            annotation = annotation.resize(image.size, Image.Resampling.NEAREST)
        canonical = _palette_mask(annotation, cancel)
        negatives = 0
        for y in range(0, image.height, TILE_SIZE):
            for x in range(0, image.width, TILE_SIZE):
                _check_cancel(cancel)
                tile = canonical[y:y + TILE_SIZE, x:x + TILE_SIZE]
                foreground = (tile > 0) & (tile < 7)
                if not foreground.any():
                    if negatives >= 2 or np.all(tile == IGNORE_INDEX):
                        continue
                    negatives += 1
                stem = f"{Path(name).stem}_{digest[:10]}_{x}_{y}"
                image_rel, mask_rel, label_rel = f"images/{split}/{stem}.jpg", f"masks/{split}/{stem}.png", f"labels/{split}/{stem}.txt"
                _save_image(stage / image_rel, image.crop((x, y, x + tile.shape[1], y + tile.shape[0])))
                (stage / mask_rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile).save(stage / mask_rel)
                lines, omitted = _polygons(tile, cancel)
                (stage / label_rel).parent.mkdir(parents=True, exist_ok=True)
                (stage / label_rel).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                present = [int(i) - 1 for i in np.unique(tile) if 1 <= i <= 6]
                class_counts.update(present)
                rows.append({"source_id": "unibo", "image": image_rel, "mask": mask_rel, "labels": label_rel,
                             "split": split, "source_split": original_split, "well_group": wells[name], "source_sha256": digest,
                             "source": entry.provenance(), "annotation": ann.provenance(), "annotation_sha256": annotation_digest,
                             "original_annotation": raw_ann.relative_to(stage).as_posix(), "original_size": [width, height],
                             "resize_factor": factor, "tile_xywh": [x, y, tile.shape[1], tile.shape[0]], "class_ids": present,
                             "ignored_pixels": int((tile == IGNORE_INDEX).sum()), "yolo_omitted_pixels": omitted, "objects": len(lines)})
    return {"task": "segmentation", "model_task": "facies", "classes": UNIBO_CLASSES,
            "split_notes": {"strategy": "whole_borehole_before_tiling", "well_assignments": assignments,
                            "original_split_overlapping_wells": sorted(w for w, s in original_groups.items() if len(s) > 1),
                            "max_side": MAX_SEGMENTATION_SIDE, "tile_size": TILE_SIZE, "mask_ignore_index": IGNORE_INDEX,
                            "palette_tolerance_rgb_euclidean": PALETTE_TOLERANCE, "palette_minimum_margin": PALETTE_MARGIN,
                            "semantic_masks_are_canonical": True, "small_yolo_components_omitted": True},
            "class_tile_counts": dict(class_counts)}, rows, issues


def _is_uv(name: str) -> bool:
    # Same convention as scripts/prepare_corebreakout_yolo.py.
    return bool(re.search(r"(?<![a-z0-9])uv(?![a-z0-9])|ультрафиолет", name.casefold()))


def _load_column_groups(path: Path) -> dict[str, str]:
    """Read the image,well_id contract without importing a training CLI."""
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "well_id"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV должен содержать столбцы image,well_id")
        for row in reader:
            name = PurePosixPath((row.get("image") or "").strip().replace("\\", "/")).name.casefold()
            well = (row.get("well_id") or "").strip().casefold()
            if not name or not well or (name in result and result[name] != well):
                raise ValueError(f"Пустая или противоречивая группа скважины для {name}")
            result[name] = well
    if not result:
        raise ValueError("CSV соответствий скважинам пуст")
    return result


def _prepare_corebreakout(inputs, stage, progress, cancel):
    mapping = {}
    for entry in inputs.entries:
        if entry.disk and entry.source == entry.disk and entry.disk.suffix.casefold() == ".csv":
            for key, value in _load_column_groups(entry.disk).items():
                if key in mapping and mapping[key] != value:
                    raise ValueError(f"Противоречивые скважины для {key}")
                mapping[key] = value
    by_source = defaultdict(dict)
    for entry in inputs.entries:
        by_source[str(entry.source)][entry.name.replace("\\", "/").casefold()] = entry
    candidates = [e for e in inputs.entries if PurePosixPath(e.name).suffix.casefold() == ".json"]
    items, issues, seen = [], [], {}
    for index, ann in enumerate(candidates, 1):
        _notify(progress, cancel, "preparing", index - 1, len(candidates), ann.name)
        if _is_uv(ann.name):
            issues.append({"reason": "ultraviolet_not_rgb", **ann.provenance()})
            continue
        ann_data = ann.read(cancel)
        try:
            obj = json.loads(ann_data.decode("utf-8-sig"))
        except (UnicodeError, ValueError):
            issues.append({"reason": "not_labelme_json", **ann.provenance()})
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("shapes"), list):
            continue
        directory = PurePosixPath(ann.name.replace("\\", "/")).parent
        names = [str(directory / (PurePosixPath(ann.name).stem + ext)) for ext in IMAGE_SUFFIXES]
        if obj.get("imagePath"):
            # LabelMe may remember a path outside the published folder. Only its
            # basename is looked up inside already indexed inputs; never open it.
            image_name = PurePosixPath(str(obj["imagePath"]).replace("\\", "/")).name
            names.insert(0, str(directory / image_name))
        entry = next((by_source[str(ann.source)].get(n.casefold()) for n in names if n.casefold() in by_source[str(ann.source)]), None)
        if entry:
            if _is_uv(entry.name):
                issues.append({"reason": "ultraviolet_not_rgb", **entry.provenance()})
                continue
            image = _decode(entry.read(cancel), entry.name)
            provenance = entry.provenance()
        elif obj.get("imageData"):
            embedded = str(obj["imageData"])
            if len(embedded) > MAX_MEMBER_BYTES * 4 // 3 + 16:
                raise ValueError(f"Слишком большое встроенное фото: {ann.name}")
            image = _decode(base64.b64decode(embedded, validate=True), ann.name)
            provenance = {**ann.provenance(), "embedded_imageData": True}
        else:
            raise ValueError(f"Не найдено фото LabelMe: {ann.name}")
        width, height = image.size
        if (int(obj.get("imageWidth") or width), int(obj.get("imageHeight") or height)) != image.size:
            raise ValueError(f"Размер LabelMe не совпадает с фото: {ann.name}")
        polygons, labels = [], []
        for shape in obj["shapes"]:
            label = str(shape.get("label") or "")
            if not label.casefold().startswith("col"):
                continue  # Match existing core-column script: never include tray.
            points = np.asarray(shape.get("points") or [], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError(f"Некорректный контур столбика: {ann.name}")
            if np.any(points < 0) or np.any(points[:, 0] > width) or np.any(points[:, 1] > height) or abs(cv2.contourArea(points.astype(np.float32))) < 4:
                raise ValueError(f"Контур столбика вне изображения или нулевой площади: {ann.name}")
            polygons.append(points)
            labels.append(label)
        digest = _pixel_hash(image)
        canonical_shapes = [(label, np.round(points, 5).tolist()) for label, points in zip(labels, polygons)]
        group = mapping.get(PurePosixPath(entry.name if entry else ann.name).name.casefold(), "")
        if mapping and not group:
            raise ValueError(f"Нет группы скважины в CSV: {ann.name}")
        fingerprint = json.dumps(canonical_shapes, sort_keys=True)
        if digest in seen:
            if seen[digest] != (fingerprint, group):
                raise ValueError(f"Копия фото с несовместимой разметкой или скважиной: {ann.name}")
            issues.append({"reason": "duplicate_photo", **provenance})
            continue
        seen[digest] = fingerprint, group
        original_size = list(image.size)
        image, _, factor = _resize_pair(image)
        temp_image = stage / ".objects" / f"{digest}.jpg"
        _save_image(temp_image, image)
        ann_rel = f"original_annotations/{digest}.json"
        (stage / ann_rel).parent.mkdir(parents=True, exist_ok=True)
        (stage / ann_rel).write_bytes(ann_data)
        items.append({"temp_image": temp_image, "original_size": original_size, "resize_factor": factor,
                      "polygons": polygons, "original_shape_labels": labels, "source": provenance,
                      "annotation": ann.provenance(), "original_annotation": ann_rel, "source_sha256": digest,
                      "well_group": group or None, "group": group or digest, "source_split": _original_split(ann.name)})
    if not items or not any(i["polygons"] for i in items):
        raise ValueError("Не найдены LabelMe-аннотации столбиков col*.")
    assignments = _group_splits({i["group"] for i in items})
    rows = []
    for index, item in enumerate(items):
        _check_cancel(cancel)
        width, height = item["original_size"]
        split = assignments[item.pop("group")]
        stem = f"core_{index:05d}_{item['source_sha256'][:12]}"
        image_rel, labels_rel = f"images/{split}/{stem}.jpg", f"labels/{split}/{stem}.txt"
        (stage / image_rel).parent.mkdir(parents=True, exist_ok=True)
        item.pop("temp_image").rename(stage / image_rel)
        lines = ["0 " + " ".join(f"{v:.8f}" for x, y in points for v in (x / width, y / height)) for points in item.pop("polygons")]
        (stage / labels_rel).parent.mkdir(parents=True, exist_ok=True)
        (stage / labels_rel).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rows.append({"source_id": "corebreakout", **item, "image": image_rel, "labels": labels_rel, "split": split,
                     "objects": len(lines)})
    (stage / ".objects").rmdir()
    warnings = [] if mapping else ["Нет подтверждённой привязки всех фото к скважинам: разделение по уникальному фото; оценка переноса на новые скважины недоступна."]
    return {"task": "segmentation", "model_task": "core_columns", "classes": [{"id": 0, "name": "core_column", "source_label": "col*", "display_ru": "Столбик керна"}],
            "warnings": warnings, "split_notes": {"strategy": "whole_well" if mapping else "photo_only", "source_splits_preserved_as_metadata": True,
            "excluded_labels": ["tray*"], "excluded_file_types": [".h5", ".pkl"], "uv_excluded": True}}, rows, issues


_RU_CLASSES = {"sandstone": "Песчаник", "limestone": "Известняк", "shale": "Глинистый сланец (shale)", "garbage": "Помехи и пустой лоток", "granite": "Гранит", "basalt": "Базальт", "mudstone": "Глинистая порода (mudstone)", "siltstone": "Алевролит", "dolomite": "Доломит", "dolostone": "Доломитовая порода", "conglomerate": "Конгломерат", "breccia": "Брекчия", "gneiss": "Гнейс", "quartzite": "Кварцит", "marble": "Мрамор"}
# Retain the project's previously sealed Mendeley well identities. The extraction
# rule is shared with vlm_core_v1/inventory_mendeley_lithology.py: before first '_'.
_MENDELEY_TEST_WELLS = {"08PD04c", "HODD2", "COD012", "SAE 6"}
_MENDELEY_VAL_WELLS = {"WC05D001", "MMDD1", "COD013", "IHAD2"}
_MENDELEY_KNOWN_WELLS = {
    "07PD01", "08PD04c", "AS10D04", "BDH 2", "CDP008", "COD011", "COD012", "COD013",
    "GHDD2", "HODD1", "HODD2", "IHAD1", "IHAD2", "IHAD3", "LB04D005", "MC07RD03",
    "MMDD1", "MURNAROO 1", "OBD 06", "OBD 7", "OOLDEA 1", "PLAYFORD 1", "PRICES BORE 1",
    "RL 1", "SAE 6", "SH 7", "WC05D001", "WDDD2",
}


def _mendeley_well(filename: str) -> str | None:
    """Recognize the audited archive's 28 wells, never arbitrary crop prefixes."""
    if "_" not in filename:
        return None
    prefix = filename.split("_", 1)[0].strip().casefold()
    return next((well for well in _MENDELEY_KNOWN_WELLS if well.casefold() == prefix), None)


def _classification_label(name: str, source_id: str) -> tuple[str, str, str]:
    parts = list(PurePosixPath(name.replace("\\", "/")).parts)
    split = _original_split(name)
    variant = next((p for p in parts if re.match(r"(?i)^(?:dcid|noise)[-_]\d+[-_]\d+", p)), source_id)
    split_words = {"train", "training", "training_subset", "val", "valid", "validation", "validation_subset", "test", "testing", "test_subset"}
    generics = {"data", "dataset", "images", "image", "imgs", "pictures", "photos", "dcid", "lithology", "lithology dataset", "test data", "training data", "validation data"}
    candidates = [p for p in parts[:-1] if p.casefold() not in split_words | generics and p != variant]
    if not candidates:
        raise ValueError(f"Не удалось определить исходный класс по папке: {name}")
    return variant, split, candidates[-1]


def _prepare_classification(source_id, inputs, stage, progress, cancel):
    candidates = [e for e in inputs.entries if _is_image(e.name)]
    rows, issues, seen, class_keys, roots = [], [], {}, {}, set()
    global_hash_subsets = defaultdict(set)
    original_well_splits = defaultdict(set)
    validated_encodings, decode_cache_hits = {}, 0
    priority = {"unspecified": 0, "train": 1, "val": 2, "test": 3}
    for index, entry in enumerate(candidates, 1):
        _notify(progress, cancel, "preparing", index - 1, len(candidates), entry.name)
        # Root overview JPEGs on HF/GitHub have no class directory and are not samples.
        try:
            variant, split, label = _classification_label(entry.name, source_id)
        except ValueError:
            issues.append({"reason": "no_source_class_directory", **entry.provenance()})
            continue
        try:
            image_data = entry.read(cancel)
        except zipfile.BadZipFile as exc:
            issues.append({"reason": "source_member_crc_error_quarantined", "error": str(exc), **entry.provenance()})
            continue
        encoding_hash = hashlib.sha256(image_data).hexdigest()
        if encoding_hash in validated_encodings:
            # The complete member was read and its ZIP CRC checked above. Only
            # identical validated encoded bytes reuse a decode; differently
            # encoded images still receive the full pixel-duplicate check.
            digest, original_size = validated_encodings[encoding_hash]
            decode_cache_hits += 1
        else:
            try:
                image = _decode(image_data, entry.name)
            except (OSError, ValueError) as exc:
                issues.append({"reason": "invalid_source_image_quarantined", "error": str(exc), **entry.provenance()})
                continue
            digest, original_size = _pixel_hash(image), list(image.size)
            validated_encodings[encoding_hash] = digest, original_size
        filename = PurePosixPath(entry.name.replace("\\", "/")).name
        well_group = _mendeley_well(filename) if source_id == "mendeley" else None
        if well_group:
            original_well_splits[well_group].add(split)
        if global_hash_subsets[digest] and variant not in global_hash_subsets[digest]:
            issues.append({"reason": "shared_pixels_between_dataset_subsets", "dataset_subset": variant,
                           "other_subsets": sorted(global_hash_subsets[digest]), **entry.provenance()})
        global_hash_subsets[digest].add(variant)
        key = variant, digest
        old = seen.get(key)
        if old is not None:
            if old["class_name"] != label or (source_id == "mendeley" and old["well_group"] != well_group):
                if old.get("retained", True):
                    (stage / old["image"]).unlink(missing_ok=True)
                old["retained"] = False
                issues.append({"reason": "conflicting_class_or_well_for_same_pixels", "previous_class": old["class_name"], "class_name": label, **entry.provenance()})
                continue
            issues.append({"reason": "duplicate_pixels", "kept_split": max((old["split"], split), key=priority.get), "previous_source": old["source"], **entry.provenance()})
            if not old.get("retained", True) or priority[split] <= priority[old["split"]]:
                continue
            (stage / old["image"]).unlink(missing_ok=True)
            old["retained"] = False
        key_class = variant, label
        if key_class not in class_keys:
            class_keys[key_class] = len(class_keys)
        root = f"classification/{_slug(variant)}"
        roots.add(root)
        suffix = PurePosixPath(entry.name.replace("\\", "/")).suffix.casefold()
        relative = f"{root}/{split}/{_slug(label)}/{digest[:24]}{suffix}"
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_data)  # validated original encoding; no extra JPEG loss
        row = {"source_id": source_id, "dataset_subset": variant, "split": split, "source_split": split, "class_name": label,
               "class_display_ru": _RU_CLASSES.get(label.casefold(), f"Исходный класс: {label}"), "class_id": class_keys[key_class],
               "image": relative, "source": entry.provenance(), "source_sha256": digest, "original_size": original_size, "well_group": well_group, "retained": True}
        rows.append(row)
        seen[key] = row
    rows = [{k: v for k, v in r.items() if k != "retained"} for r in rows if r["retained"]]
    if not rows:
        raise ValueError("Не найдены изображения в исходных папках классов. Pickle и модели не исполняются.")
    well_assignments = {}
    unknown_mendeley_wells = source_id == "mendeley" and any(not r["well_group"] for r in rows)
    if source_id == "mendeley" and not unknown_mendeley_wells:
        wells = {r["well_group"] for r in rows}
        if wells & (_MENDELEY_TEST_WELLS | _MENDELEY_VAL_WELLS):
            well_assignments = {well: "test" if well in _MENDELEY_TEST_WELLS else "val" if well in _MENDELEY_VAL_WELLS else "train" for well in wells}
            # Small selections may contain only part of the published wells.
            available = sorted((w for w in wells if well_assignments[w] == "train"), key=lambda w: hashlib.sha256(w.encode()).hexdigest())
            if "val" not in well_assignments.values() and len(available) >= 2:
                well_assignments[available.pop(0)] = "val"
            if "test" not in well_assignments.values() and len(available) >= 2:
                well_assignments[available.pop(0)] = "test"
        else:
            well_assignments = _group_splits(wells)
        for index, row in enumerate(rows):
            if index % 1000 == 0:
                _notify(progress, cancel, "partitioning", index, len(rows), "Разделение по целым скважинам")
            new_split = well_assignments[row["well_group"]]
            if new_split != row["split"]:
                previous = stage / row["image"]
                parts = list(PurePosixPath(row["image"]).parts)
                parts[2] = new_split
                row["image"], row["split"] = PurePosixPath(*parts).as_posix(), new_split
                (stage / row["image"]).parent.mkdir(parents=True, exist_ok=True)
                previous.rename(stage / row["image"])
    # The source's final test stays untouched. Where no validation subset was
    # published, reserve deterministic hashes from its original training set.
    variants = sorted({r["dataset_subset"] for r in rows})
    for index, variant in enumerate(variants):
        _notify(progress, cancel, "partitioning", index, len(variants), "Проверка независимых частей набора")
        if source_id == "mendeley":
            continue  # Source splits are known to share wells; never crop-split.
        subset = [r for r in rows if r["dataset_subset"] == variant]
        if any(r["split"] == "val" for r in subset):
            continue
        by_class = defaultdict(list)
        for row in subset:
            if row["split"] == "train":
                by_class[row["class_name"]].append(row)
        for candidates_for_class in by_class.values():
            ordered = sorted(candidates_for_class, key=lambda r: r["source_sha256"])
            n_val = max(1, round(len(ordered) * .15)) if len(ordered) >= 2 else 0
            for row in ordered[:n_val]:
                previous = stage / row["image"]
                parts = list(PurePosixPath(row["image"]).parts)
                parts[2] = "val"
                row["image"] = PurePosixPath(*parts).as_posix()
                row["split"] = "val"
                (stage / row["image"]).parent.mkdir(parents=True, exist_ok=True)
                previous.rename(stage / row["image"])
    retained_keys = {(r["dataset_subset"], r["class_name"]) for r in rows}
    classes = [{"id": class_keys[k], "dataset_subset": k[0], "name": k[1], "folder_name": _slug(k[1]),
                "display_ru": _RU_CLASSES.get(k[1].casefold(), f"Исходный класс: {k[1]}")} for k in sorted(retained_keys)]
    return {"task": "classification", "model_task": "lithology_classification", "classes": classes, "classification_relative_roots": sorted(roots),
            "decoded_unique_encodings": len(validated_encodings), "decode_cache_hits": decode_cache_hits,
            "subset_counts": {variant: dict(Counter(r["split"] for r in rows if r["dataset_subset"] == variant)) for variant in sorted({r["dataset_subset"] for r in rows})},
            "requires_independent_validation": unknown_mendeley_wells,
            "split_notes": {"strategy": "whole_well_reusing_project_holdouts" if well_assignments else "preserve_source_test; reserve_validation_from_source_train_when_missing", "duplicate_policy": "retain_one_per_subset_prefer_test_then_val_then_train; quarantine_conflicting_labels_and_mendeley_crosswell_copies",
                            "well_disjointness": "verified_filename_prefix" if well_assignments else "not_verified", "well_assignments": well_assignments,
                            "well_counts": dict(Counter(well_assignments.values())),
                            "original_split_overlapping_wells": sorted(w for w, splits in original_well_splits.items() if len(splits) > 1),
                            "original_image_encoding_preserved": True, "no_descriptions_generated": True},
            "warnings": ["Классы пород сохранены из источника. Подробных описаний и автоматического соответствия справочнику фаций нет."] +
                        (["Не удалось прочитать скважины Mendeley из имён файлов. Исходные части могут содержать одни скважины; обучение заблокировано до независимого разделения."] if unknown_mendeley_wells else [])}, rows, issues


def prepare_dataset(source_id: str, source_paths: list[Path], output_dir: Path, *, progress: Callable[[dict], None] | None = None, cancel=None) -> dict:
    """Prepare audited local files, preserving source names, rights and provenance.

    Supported source IDs: unibo, corebreakout, dcid, mendeley. The optional
    CoreBreakout CSV uses the existing script's columns ``image,well_id``.
    Never overwrites a destination. ``readiness`` describes file/split readiness,
    not geological approval, commercial permission, or achieved model accuracy.
    """
    aliases = {"unibo_facies_7308": "unibo", "bologna": "unibo", "core_breakout": "corebreakout", "mendeley_lithology": "mendeley"}
    source_id = aliases.get(source_id.casefold(), source_id.casefold())
    if source_id not in SOURCE_INFO:
        raise ValueError(f"Неизвестный открытый набор: {source_id}")
    if any(Path(p).is_symlink() for p in source_paths):
        raise ValueError("Символические ссылки не поддерживаются.")
    paths = [Path(p).resolve() for p in source_paths]
    if not paths:
        raise ValueError("Не выбраны исходные файлы.")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Папка результата уже существует: {output_dir}")
    for path in paths:
        if path.is_dir() and (output_dir == path or path in output_dir.parents):
            raise ValueError("Результат должен находиться вне выбранной исходной папки.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.preparing-", dir=output_dir.parent))
    try:
        _notify(progress, cancel, "indexing", 0, len(paths), "Проверка исходных архивов")
        with _Inputs(paths, cancel) as inputs:
            if source_id == "unibo":
                extra, rows, issues = _prepare_unibo(inputs, stage, progress, cancel)
            elif source_id == "corebreakout":
                extra, rows, issues = _prepare_corebreakout(inputs, stage, progress, cancel)
            else:
                extra, rows, issues = _prepare_classification(source_id, inputs, stage, progress, cancel)
            _check_cancel(cancel)
            if not rows:
                raise ValueError("Подготовка не создала пригодных примеров.")
            counts = dict(Counter(r["split"] for r in rows))
            task = extra["task"]
            segmentation = task == "segmentation"
            if segmentation:
                for split in ("train", "val", "test"):
                    (stage / "images" / split).mkdir(parents=True, exist_ok=True)
                    (stage / "labels" / split).mkdir(parents=True, exist_ok=True)
                lines = ["# Source-specific classes; paths resolve relative to this YAML file.", "train: images/train", "val: images/val"]
                if counts.get("test"):
                    lines.append("test: images/test")
                lines += ["names:"] + [f"  {c['id']}: {json.dumps(c['name'], ensure_ascii=False)}" for c in extra["classes"]]
                (stage / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
            positive_counts = dict(Counter(r["split"] for r in rows if r.get("objects", 1) > 0))
            has_evaluation = positive_counts.get("val", 0)
            summary = {**extra, "source_id": source_id, "output_dir": str(output_dir), "data_yaml": str(output_dir / "data.yaml") if segmentation else None,
                       "classification_roots": [str(output_dir / p) for p in extra.get("classification_relative_roots", [])],
                       "data_yaml_relative": "data.yaml" if segmentation else None, "manifest_relative": "manifest.jsonl",
                       "manifest": str(output_dir / "manifest.jsonl"), "counts": counts, "sample_count": len(rows),
                       "positive_counts": positive_counts,
                       "readiness": "ready" if positive_counts.get("train") and has_evaluation and not extra.get("requires_independent_validation") else "needs_validation",
                       "source_catalog_only": True, "geologist_approved": False, "source": SOURCE_INFO[source_id],
                       "license": SOURCE_INFO[source_id]["license"], "issue_count": len(issues), "sources": inputs.sources,
                       "warnings": extra.get("warnings", [])}
            if summary["readiness"] != "ready":
                summary["warnings"].append("Нет независимой обучающей и проверочной частей; обучение из интерфейса пока недоступно.")
            _write_json(stage / "classes.json", extra["classes"])
            _write_rows(stage / "manifest.jsonl", rows)
            _write_rows(stage / "issues.jsonl", issues)
            _write_json(stage / "summary.json", summary)
            (stage / "LICENSE_SOURCES.md").write_text("# Source and attribution\n\n" + "\n\n".join(f"{key}: {value}" for key, value in SOURCE_INFO[source_id].items()) + "\n\nSource labels are preserved. Check permission for the intended use before distributing trained models.\n", encoding="utf-8")
        _notify(progress, cancel, "finalizing", len(rows), len(rows), "Сохранение подготовленного набора")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        stage.rename(output_dir)
        return summary
    except BaseException:
        # The staging path is created directly under the verified target parent.
        if stage.parent == output_dir.parent and stage.name.startswith(f".{output_dir.name}.preparing-"):
            shutil.rmtree(stage, ignore_errors=True)
        raise
