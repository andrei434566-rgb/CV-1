"""Independent stage-1 recognition and training for physical core columns.

Facies are deliberately absent from this module.  Its only class is a physical
``core_column``.  A dedicated YOLO model is used when one has been trained; the
existing deterministic detector remains an offline bootstrap/fallback.  The
result is therefore available before the facies model is invoked.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from app.domain.models import PhotoRecord
from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector
from app.runtime_paths import bundled_root, user_data_root


MIN_COLUMN_TRAINING_PHOTOS = 3
COLUMN_CLASS_NAME = "core_column"
_MODEL_CACHE: dict[Path, object] = {}


@dataclass(frozen=True)
class CoreTapeSection:
    """Mapping between one source rectangle and its place in a vertical tape."""

    source_left: int
    source_top: int
    source_right: int
    source_bottom: int
    tape_top: int
    tape_bottom: int


@dataclass(frozen=True)
class AssembledCoreTape:
    """All detected columns concatenated vertically without spacer pixels."""

    image: np.ndarray
    sections: tuple[CoreTapeSection, ...]


def read_bgr_image(image_path: str | Path) -> np.ndarray:
    """Read a color image reliably from Windows paths containing Cyrillic."""
    path = Path(image_path)
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать изображение: {path.name}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {path.name}")
    return image


def normalize_columns(
    columns: Iterable[dict[str, float] | tuple[int, int, int, int]],
    image_size: tuple[int, int],
    *,
    suppress_narrow_shadows: bool = False,
) -> list[dict[str, float]]:
    """Clamp, de-duplicate and order physical column rectangles."""
    width, height = image_size
    candidates: list[dict[str, float]] = []
    for values in columns:
        try:
            if isinstance(values, dict):
                left, top, right, bottom = (float(values[key]) for key in ("left", "top", "right", "bottom"))
            else:
                left, top, right, bottom = (float(value) for value in values)
        except (KeyError, TypeError, ValueError):
            continue
        left, right = max(0.0, min(left, width)), max(0.0, min(right, width))
        top, bottom = max(0.0, min(top, height)), max(0.0, min(bottom, height))
        if right - left < 2 or bottom - top < 2:
            continue
        candidate = {"left": left, "top": top, "right": right, "bottom": bottom}
        candidates.append(candidate)
    # Automatic proposals can include a thin strip inside the actual lane.
    # Keep interpreter-confirmed rectangles as drawn unless this specific
    # suppression is requested by the automatic recognition path.
    candidates.sort(
        key=lambda item: (
            item["right"] - item["left"],
            (item["right"] - item["left"]) * (item["bottom"] - item["top"]),
        ),
        reverse=True,
    )
    boxes: list[dict[str, float]] = []
    for candidate in candidates:
        if any(
            _rectangle_iou(candidate, existing) >= 0.72
            or (suppress_narrow_shadows and _is_narrow_shadow(candidate, existing))
            for existing in boxes
        ):
            continue
        boxes.append(candidate)
    return sorted(boxes, key=lambda item: (item["left"], item["top"]))


def _merge_column_candidates(
    columns: Iterable[dict[str, float] | tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> list[dict[str, float]]:
    """Fuse overlapping detector proposals that refer to the same x-lane.

    The neural detector and image heuristics often return different portions of
    the same half/fragment. Plain IoU suppression incorrectly keeps both boxes
    (or drops one); lane-based fusion takes their envelope before deduplication.
    """
    width, height = image_size
    normalized = normalize_columns(columns, image_size, suppress_narrow_shadows=True)
    merged: list[dict[str, float]] = []
    for box in normalized:
        box_width = box["right"] - box["left"]
        box_center = (box["left"] + box["right"]) / 2
        match_index = None
        for index, current in enumerate(merged):
            current_width = current["right"] - current["left"]
            overlap = min(box["right"], current["right"]) - max(box["left"], current["left"])
            smaller_width = min(box_width, current_width)
            current_center = (current["left"] + current["right"]) / 2
            if (
                smaller_width > 0
                and overlap >= smaller_width * 0.55
                and abs(box_center - current_center) <= max(5.0, smaller_width * 0.48)
            ):
                match_index = index
                break
        if match_index is None:
            merged.append(dict(box))
            continue
        current = merged[match_index]
        current.update({
            "left": max(0.0, min(current["left"], box["left"])),
            "top": max(0.0, min(current["top"], box["top"])),
            "right": min(float(width), max(current["right"], box["right"])),
            "bottom": min(float(height), max(current["bottom"], box["bottom"])),
        })
    return normalize_columns(merged, image_size, suppress_narrow_shadows=True)


def assemble_core_tape(
    image: np.ndarray,
    columns: Iterable[dict[str, float] | tuple[int, int, int, int]],
    target_width: int | None = None,
) -> AssembledCoreTape:
    """Crop and concatenate columns into one continuous vertical image.

    No separator or padding is inserted between adjacent source columns.  The
    returned section map makes the operation auditable and reversible.
    """
    if image is None or image.size == 0 or image.ndim != 3:
        raise ValueError("Пустое или некорректное изображение.")
    height, width = image.shape[:2]
    boxes = normalize_columns(columns, (width, height))
    if not boxes:
        raise ValueError("Нет столбиков керна для сборки единой колонки.")
    widths = [max(1, round(item["right"] - item["left"])) for item in boxes]
    tape_width = max(1, int(target_width or round(float(np.median(widths)))))
    crops: list[np.ndarray] = []
    sections: list[CoreTapeSection] = []
    cursor = 0
    for box in boxes:
        left, top, right, bottom = (
            int(round(box["left"])), int(round(box["top"])),
            int(round(box["right"])), int(round(box["bottom"])),
        )
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        scaled_height = max(1, round(crop.shape[0] * tape_width / max(1, crop.shape[1])))
        interpolation = cv2.INTER_AREA if tape_width < crop.shape[1] else cv2.INTER_CUBIC
        normalized = cv2.resize(crop, (tape_width, scaled_height), interpolation=interpolation)
        crops.append(normalized)
        sections.append(CoreTapeSection(left, top, right, bottom, cursor, cursor + scaled_height))
        cursor += scaled_height
    if not crops:
        raise ValueError("Выделенные столбики керна пусты.")
    return AssembledCoreTape(np.concatenate(crops, axis=0), tuple(sections))


class CoreColumnRecognizer:
    """Recognize physical core columns independently of facies inference."""

    def __init__(self, model_path: str | Path | None = None, confidence: float = 0.25, image_size: int = 768):
        selected_model = Path(model_path) if model_path else self.default_model_path()
        self.model_path = selected_model.expanduser().resolve() if selected_model else None
        self.confidence = max(0.01, min(0.99, float(confidence)))
        self.image_size = max(320, min(1536, int(image_size)))
        self.model = None
        self.device: int | str = "cpu"
        self.source_label = "детерминированный детектор"
        if self.model_path and self.model_path.is_file():
            from ultralytics import YOLO

            candidate = _MODEL_CACHE.get(self.model_path)
            if candidate is None:
                candidate = YOLO(str(self.model_path))
                _MODEL_CACHE[self.model_path] = candidate
            names = {str(value).strip().casefold() for value in dict(getattr(candidate.model, "names", {}) or {}).values()}
            if names & {COLUMN_CLASS_NAME, "core", "core column", "core columns", "керн", "столбик керна"}:
                self.model = candidate
                self.device = self._best_device()
                self.source_label = f"модель {self.model_path.name}"

    @staticmethod
    def default_model_path() -> Path | None:
        bundled = bundled_root() / "models" / "core_columns" / "best.pt"
        if bundled.is_file():
            return bundled
        trained = sorted(
            (path for path in (user_data_root() / "models" / "core_columns").glob("*/best.pt") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return trained[0] if trained else None

    def recognize(self, image: np.ndarray, target_size: tuple[int, int] | None = None) -> list[dict[str, float]]:
        if image is None or image.size == 0:
            raise ValueError("Пустое изображение.")
        source_height, source_width = image.shape[:2]
        model_boxes = self._model_boxes(image) if self.model is not None else []
        heuristic_boxes = RuleBasedFaciesDetector._find_core_columns(image)
        boxes = _merge_column_candidates((*model_boxes, *heuristic_boxes), (source_width, source_height))
        if model_boxes and heuristic_boxes:
            self.source_label = f"{self.model_path.name if self.model_path else 'модель'} + резервный поиск"
        elif model_boxes:
            self.source_label = f"модель {self.model_path.name if self.model_path else ''}".strip()
        else:
            self.source_label = "детерминированный детектор"
        normalized = normalize_columns(boxes, (source_width, source_height), suppress_narrow_shadows=True)
        # Scanned plates may contain a narrow ruler and short arrows next to
        # several full-length core lanes. They are useful image features, but
        # they must not become physical core columns.
        if not model_boxes:
            long_lanes = [
                box for box in normalized
                if box["bottom"] - box["top"] >= source_height * 0.45
                and box["right"] - box["left"] >= source_width * 0.05
            ]
            if len(long_lanes) >= 2:
                typical_width = float(np.median([box["right"] - box["left"] for box in long_lanes]))
                typical_height = float(np.median([box["bottom"] - box["top"] for box in long_lanes]))

                def is_core_or_substantial_fragment(box: dict[str, float]) -> bool:
                    width = box["right"] - box["left"]
                    height = box["bottom"] - box["top"]
                    if width >= typical_width * 0.55:
                        return True
                    if height < typical_height * 0.25:
                        return False
                    crop = image[int(box["top"]):int(box["bottom"]), int(box["left"]):int(box["right"])]
                    if crop.size == 0:
                        return False
                    dark_fraction = np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 180)
                    return bool(dark_fraction >= 0.45)

                normalized = [box for box in normalized if is_core_or_substantial_fragment(box)]
        # Suppress obvious false positives beside a set of consistent, full-length
        # core lanes (rulers, paper labels and background objects).  Do not
        # apply this to a single lane or to manually confirmed rectangles.
        reference = [
            box for box in normalized
            if (box["bottom"] - box["top"]) >= source_height * 0.45
            and (box["bottom"] - box["top"]) >= 1.7 * (box["right"] - box["left"])
        ]
        if len(reference) >= 2:
            reference_heights = np.array([b["bottom"] - b["top"] for b in reference])
            reference_widths = np.array([b["right"] - b["left"] for b in reference])
            median_height = float(np.median(reference_heights))
            median_width = float(np.median(reference_widths))
            strong = [
                b for b in reference
                if b["bottom"] - b["top"] >= median_height * 0.65
                and b["right"] - b["left"] >= median_width * 0.55
            ]
            if len(strong) >= 2:
                left_edge = min(b["left"] for b in strong)
                right_edge = max(b["right"] for b in strong)
                filtered = []
                for box in normalized:
                    bw = box["right"] - box["left"]
                    bh = box["bottom"] - box["top"]
                    cx = (box["left"] + box["right"]) / 2
                    outside = cx < left_edge - median_width * 0.35 or cx > right_edge + median_width * 0.35
                    short = bh < median_height * 0.55
                    thin = bw < median_width * 0.48
                    if (outside and (short or thin)) or (thin and bh >= median_height * 0.7):
                        continue
                    filtered.append(box)
                normalized = filtered
        if target_size is None:
            return normalized
        x_scale = target_size[0] / max(1, source_width)
        y_scale = target_size[1] / max(1, source_height)
        return [
            {
                "left": item["left"] * x_scale,
                "top": item["top"] * y_scale,
                "right": item["right"] * x_scale,
                "bottom": item["bottom"] * y_scale,
            }
            for item in normalized
        ]

    def recognize_path(self, image_path: str | Path, target_size: tuple[int, int] | None = None) -> list[dict[str, float]]:
        return self.recognize(read_bgr_image(image_path), target_size)

    def _model_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        results = self.model(
            image,
            verbose=False,
            device=self.device,
            half=self.device != "cpu",
            imgsz=self.image_size,
            conf=self.confidence,
            max_det=64,
        )
        height, width = image.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        for result in results:
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None:
                continue
            for box in result_boxes:
                left, top, right, bottom = box.xyxy[0].tolist()
                box_width, box_height = right - left, bottom - top
                if (
                    box_width >= max(5, width * 0.006)
                    and box_width <= width * 0.76
                    and box_height >= max(16, height * 0.008)
                    and box_height / max(1.0, box_width) >= 0.38
                ):
                    boxes.append((round(left), round(top), round(right), round(bottom)))
        return boxes

    @staticmethod
    def _best_device() -> int | str:
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except (ImportError, RuntimeError):
            return "cpu"


def export_core_column_dataset(records: list[PhotoRecord], destination: Path) -> dict[str, object]:
    """Export interpreter-confirmed stage-1 rectangles as YOLO segments."""
    samples = [record for record in records if record.core_columns_verified and record.core_columns]
    if len(samples) < MIN_COLUMN_TRAINING_PHOTOS:
        raise ValueError(
            f"Для обучения столбиков нужно подтвердить границы минимум на {MIN_COLUMN_TRAINING_PHOTOS} фото."
        )
    if destination.exists():
        raise FileExistsError(f"Папка датасета уже существует: {destination}")
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
    validation_count = max(1, round(len(samples) * 0.2))
    split_at = len(samples) - validation_count
    rectangles = 0
    for index, record in enumerate(samples):
        split = "train" if index < split_at else "val"
        stem = f"column_{index + 1:05d}"
        image_path = destination / "images" / split / f"{stem}.jpg"
        if not record.pixmap.save(str(image_path), "JPG", 100):
            raise RuntimeError(f"Не удалось сохранить фото датасета: {Path(record.path).name}")
        width, height = record.pixmap.width(), record.pixmap.height()
        boxes = normalize_columns(record.core_columns, (width, height))
        lines = []
        for box in boxes:
            points = (
                (box["left"], box["top"]), (box["right"], box["top"]),
                (box["right"], box["bottom"]), (box["left"], box["bottom"]),
            )
            coordinates = " ".join(
                f"{max(0.0, min(1.0, x / width)):.6f} {max(0.0, min(1.0, y / height)):.6f}"
                for x, y in points
            )
            lines.append(f"0 {coordinates}")
        (destination / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rectangles += len(lines)
    yaml_path = destination / "data.yaml"
    yaml_path.write_text(
        "\n".join((
            f"path: {json.dumps(destination.as_posix(), ensure_ascii=False)}",
            "train: images/train", "val: images/val", "nc: 1", "names:",
            f"  0: {COLUMN_CLASS_NAME}", "",
        )),
        encoding="utf-8",
    )
    summary = {"photos": len(samples), "rectangles": rectangles, "data_yaml": str(yaml_path)}
    (destination / "dataset_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**summary, "data_yaml": yaml_path, "output_dir": destination}


class CoreColumnTrainingWorker(QObject):
    """Train and publish the independent one-class stage-1 model."""

    progress = Signal(str)
    epoch_progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, base_model: Path, data_yaml: Path, runs_dir: Path, published_dir: Path, epochs: int = 30):
        super().__init__()
        self.base_model = Path(base_model)
        self.data_yaml = Path(data_yaml)
        self.runs_dir = Path(runs_dir)
        self.published_dir = Path(published_dir)
        self.epochs = max(1, int(epochs))

    @Slot()
    def run(self) -> None:
        try:
            device = CoreColumnRecognizer._best_device()
            if device == "cpu":
                raise RuntimeError("Обучение распознавателя столбиков требует NVIDIA GPU с CUDA.")
            from ultralytics import YOLO

            self.runs_dir.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(self.base_model))
            model.add_callback("on_train_epoch_end", self._report_epoch)
            self.progress.emit(f"Обучение столбиков: {self.epochs} эпох")
            result = model.train(
                data=str(self.data_yaml), epochs=self.epochs, imgsz=640, device=device,
                batch=-1, cache=False, amp=True, project=str(self.runs_dir),
                name="core_columns", exist_ok=True, verbose=False,
            )
            save_dir = Path(str(getattr(result, "save_dir", self.runs_dir / "core_columns")))
            best = save_dir / "weights" / "best.pt"
            if not best.is_file():
                raise RuntimeError("Обучение завершилось без best.pt.")
            if self.published_dir.exists():
                raise FileExistsError(f"Папка модели уже существует: {self.published_dir}")
            self.published_dir.mkdir(parents=True)
            published = self.published_dir / "best.pt"
            shutil.copy2(best, published)
            shutil.copy2(self.data_yaml, self.published_dir / "data.yaml")
            (self.published_dir / "training_info.json").write_text(
                json.dumps({
                    "module": "core-column-recognition",
                    "class": COLUMN_CLASS_NAME,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "source_model": str(self.base_model),
                    "epochs": self.epochs,
                    "dataset": str(self.data_yaml.parent),
                    "run": str(save_dir),
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.succeeded.emit(str(published))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _report_epoch(self, trainer) -> None:
        total = max(1, int(getattr(trainer, "epochs", self.epochs)))
        current = max(0, min(total, int(getattr(trainer, "epoch", -1)) + 1))
        self.epoch_progress.emit(current, total)


def _rectangle_iou(first: dict[str, float], second: dict[str, float]) -> float:
    left, top = max(first["left"], second["left"]), max(first["top"], second["top"])
    right, bottom = min(first["right"], second["right"]), min(first["bottom"], second["bottom"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first["right"] - first["left"]) * (first["bottom"] - first["top"])
    second_area = (second["right"] - second["left"]) * (second["bottom"] - second["top"])
    return intersection / max(1.0, first_area + second_area - intersection)


def _is_narrow_shadow(candidate: dict[str, float], existing: dict[str, float]) -> bool:
    """Identify a thin tray rail inside or directly beside a full core lane."""
    candidate_width = candidate["right"] - candidate["left"]
    existing_width = existing["right"] - existing["left"]
    if candidate_width > existing_width * 0.55:
        return False
    candidate_height = candidate["bottom"] - candidate["top"]
    existing_height = existing["bottom"] - existing["top"]
    if existing_height < candidate_height * 0.5:
        return False
    overlap_x = min(candidate["right"], existing["right"]) - max(candidate["left"], existing["left"])
    overlap_y = min(candidate["bottom"], existing["bottom"]) - max(candidate["top"], existing["top"])
    if overlap_y < candidate_height * 0.6:
        return False
    if overlap_x >= candidate_width * 0.5:
        return True
    # For example, 16.png has a 470–481 px cardboard divider touching the
    # real 480–543 px core.  It is almost full height and has little IoU.
    if candidate_width > existing_width * 0.35 or candidate_height < existing_height * 0.7:
        return False
    gap = max(0.0, candidate["left"] - existing["right"], existing["left"] - candidate["right"])
    return gap <= max(2.0, existing_width * 0.08)
