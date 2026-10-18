"""Clean-up and gap-free projection of facies inside physical core columns."""

from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF

from app.domain.facies_catalog import facies_identity
from app.domain.models import FaciesDetection, PhotoRecord

UNRECOGNIZED_FACIES = "Не определена"


def has_complete_core_column_coverage(record: PhotoRecord, tolerance: float = 1.0) -> bool:
    """Check whether every physical core column is represented in the tape."""
    columns = [_valid_column(item) for item in record.core_columns]
    columns = [item for item in columns if item is not None]
    if not columns:
        return True
    for column in columns:
        spans: list[tuple[float, float]] = []
        for detection in record.detections:
            if detection.label == "Новый контур" or len(detection.polygon) < 3:
                continue
            bounds = QPolygonF(detection.polygon).boundingRect()
            overlap = max(0.0, min(bounds.right(), column["right"]) - max(bounds.left(), column["left"]))
            if overlap < 0.15 * (column["right"] - column["left"]):
                continue
            top = max(bounds.top(), column["top"])
            bottom = min(bounds.bottom(), column["bottom"])
            if bottom > top:
                spans.append((top, bottom))
        covered_to = column["top"]
        for top, bottom in sorted(spans):
            if top > covered_to + tolerance:
                return False
            covered_to = max(covered_to, bottom)
        if covered_to < column["bottom"] - tolerance:
            return False
    return True


def postprocess_detections(
    record: PhotoRecord,
    detections: list[FaciesDetection],
    fallback_label: str | None = None,
) -> list[FaciesDetection]:
    """Remove tiny fragments and make each detected core column gap-free.

    Reviewed contours are never rewritten.  All decisions are constrained to a
    single physical core column, preventing boundaries from being smoothed
    through a ruler, tray or a neighbouring column.  Fresh automatic results
    are finally converted to a vertical partition: every selected core pixel
    belongs to exactly one facies band.
    """
    if record.pixmap.isNull():
        return detections
    min_height = max(5.0, record.pixmap.height() * 0.002)
    # A whole photograph is not a core column: it may contain rulers, labels
    # and empty tray space.  Do not manufacture a facies tape if stage 1 failed.
    columns = record.core_columns
    if not columns:
        return list(detections)
    automatic: list[tuple[int, FaciesDetection]] = []
    retained: list[FaciesDetection] = []
    unnamed_manual: list[FaciesDetection] = []
    for detection in detections:
        if detection.label == "Новый контур":
            unnamed_manual.append(detection)
            continue
        rect = QPolygonF(detection.polygon).boundingRect()
        manually_drawn = detection.attributes.get("__manual_contour") == "1"
        if detection.training_ready or manually_drawn:
            retained.append(detection)
        elif rect.height() >= min_height:
            automatic.append((_column_for(rect, columns), detection))
    automatic.sort(key=lambda item: (item[0], QPolygonF(item[1].polygon).boundingRect().top()))
    for column_index, detection in automatic:
        previous_index = next((index for index in range(len(retained) - 1, -1, -1)
                               if not retained[index].training_ready
                               and retained[index].attributes.get("__manual_contour") != "1"
                               and _column_for(QPolygonF(retained[index].polygon).boundingRect(), columns) == column_index), None)
        if previous_index is not None and _can_merge(retained[previous_index], detection):
            retained[previous_index] = _merged(retained[previous_index], detection)
        else:
            retained.append(detection)
    preserved = [item for item in retained if item.training_ready or item.attributes.get("__manual_contour") == "1"]
    preserved_ids = {id(item) for item in preserved}
    automatic = [item for item in retained if id(item) not in preserved_ids]
    covered = complete_core_column_coverage(
        automatic,
        columns,
        fallback_label=fallback_label,
        depth_segments=record.depth_segments,
        preserved_detections=preserved,
    )
    return preserved + unnamed_manual + covered


def complete_core_column_coverage(
    detections: list[FaciesDetection],
    columns: list[dict[str, float]],
    fallback_label: str | None = None,
    depth_segments: list[dict[str, float]] | None = None,
    preserved_detections: list[FaciesDetection] | None = None,
) -> list[FaciesDetection]:
    """Turn arbitrary model masks into one gap-free partition per core column.

    Existing predictions define all candidate contacts. Uncovered intervals
    remain geometrically filled but explicitly unrecognized. Adjacency or an
    arbitrary first model class is not evidence of geological identity.
    ``fallback_label`` is retained for call compatibility, never used to guess.
    """
    valid_columns = [_valid_column(item) for item in columns]
    valid_columns = [item for item in valid_columns if item is not None]
    if not valid_columns:
        return list(detections)
    preserved = list(preserved_detections or [])
    preserved_ids = {id(item) for item in preserved}
    valid_detections = [
        item for item in [*detections, *preserved]
        if str(item.label or "").strip() not in {"", "Новый контур"} and len(item.polygon) >= 3
    ]
    result: list[FaciesDetection] = []
    for column_index, column in enumerate(valid_columns):
        candidates: list[tuple[float, float, float, FaciesDetection]] = []
        for detection in valid_detections:
            rect = QPolygonF(detection.polygon).boundingRect()
            horizontal_overlap = max(0.0, min(rect.right(), column["right"]) - max(rect.left(), column["left"]))
            if horizontal_overlap <= 0:
                continue
            top = max(column["top"], rect.top())
            bottom = min(column["bottom"], rect.bottom())
            if bottom - top <= 0.25:
                continue
            width_fraction = horizontal_overlap / max(1.0, column["right"] - column["left"])
            if width_fraction >= 0.15 or id(detection) in preserved_ids:
                candidates.append((top, bottom, width_fraction, detection))

        boundaries = {column["top"], column["bottom"]}
        for top, bottom, _, _ in candidates:
            boundaries.update((top, bottom))
        edges = sorted(boundaries)
        bands: list[FaciesDetection] = []
        for top, bottom in zip(edges[:-1], edges[1:]):
            if bottom - top <= 1e-6:
                continue
            midpoint = (top + bottom) / 2
            active = [item for item in candidates if item[0] <= midpoint < item[1]]
            gap_filled = not active
            if active:
                # Interval crops are classified at higher effective resolution
                # than the compressed full photograph, so they own the label
                # inside their structural package when available.
                chosen = max(
                    active,
                    key=lambda item: (
                        id(item[3]) in preserved_ids,
                        bool(item[3].attributes.get("__structural_interval")),
                        item[3].confidence * item[2],
                        item[2],
                        item[3].confidence,
                    ),
                )[3]
            else:
                chosen = FaciesDetection(UNRECOGNIZED_FACIES, 0.0, [], attributes={
                    "Статус фации": "нет прогноза, требуется разметка",
                })
            # Keep the interpreter's exact original polygon for training and
            # editing.  Only the uncovered spans become generated bands.
            if id(chosen) in preserved_ids:
                continue
            attributes = dict(chosen.attributes)
            attributes["Покрытие керна"] = (
                "интервал без достоверного прогноза"
                if gap_filled or chosen.label == UNRECOGNIZED_FACIES
                else "прогноз модели"
            )
            depth_from, depth_to = _band_depths(column, top, bottom, depth_segments or [])
            confidence = max(0.0, min(1.0, chosen.confidence * (0.5 if gap_filled else 1.0)))
            band = FaciesDetection(
                label=chosen.label,
                confidence=confidence,
                polygon=[
                    QPointF(column["left"], top), QPointF(column["right"], top),
                    QPointF(column["right"], bottom), QPointF(column["left"], bottom),
                ],
                attributes=attributes,
                depth_from=depth_from,
                depth_to=depth_to,
                alternatives=dict(chosen.alternatives),
            )
            if bands and _same_band(bands[-1], band):
                bands[-1] = _merged(bands[-1], band)
            else:
                bands.append(band)
        result.extend(bands)
    return result


def _valid_column(values: dict[str, float]) -> dict[str, float] | None:
    try:
        column = {key: float(values[key]) for key in ("left", "top", "right", "bottom")}
    except (KeyError, TypeError, ValueError):
        return None
    return column if column["right"] > column["left"] and column["bottom"] > column["top"] else None


def _same_band(first: FaciesDetection, second: FaciesDetection) -> bool:
    first_interval = first.attributes.get("__structural_interval")
    second_interval = second.attributes.get("__structural_interval")
    same_structure = not (first_interval or second_interval) or first_interval == second_interval
    same_identity = facies_identity(first.label, first.attributes) == facies_identity(second.label, second.attributes)
    same_status = first.attributes.get("Статус фации") == second.attributes.get("Статус фации")
    return same_identity and same_status and same_structure and abs(QPolygonF(first.polygon).boundingRect().bottom() - QPolygonF(second.polygon).boundingRect().top()) <= 1e-6


def _band_depths(
    column: dict[str, float],
    top: float,
    bottom: float,
    depth_segments: list[dict[str, float]],
) -> tuple[float | None, float | None]:
    center_x = (column["left"] + column["right"]) / 2
    segment = next(
        (
            item for item in depth_segments
            if all(key in item for key in ("left", "top", "right", "bottom", "depth_from", "depth_to"))
            and float(item["left"]) <= center_x <= float(item["right"])
        ),
        None,
    )
    if segment is None:
        return None, None
    pixel_span = float(segment["bottom"]) - float(segment["top"])
    if pixel_span <= 0:
        return None, None
    depth_span = float(segment["depth_to"]) - float(segment["depth_from"])
    start = float(segment["depth_from"]) + (top - float(segment["top"])) / pixel_span * depth_span
    end = float(segment["depth_from"]) + (bottom - float(segment["top"])) / pixel_span * depth_span
    return round(min(start, end), 4), round(max(start, end), 4)


def _column_for(rect, columns: list[dict[str, float]]) -> int:
    center_x = rect.center().x()
    for index, column in enumerate(columns):
        if float(column.get("left", 0)) <= center_x <= float(column.get("right", 0)):
            return index
    return -1


def _can_merge(first: FaciesDetection, second: FaciesDetection) -> bool:
    if facies_identity(first.label, first.attributes) != facies_identity(second.label, second.attributes):
        return False
    if first.attributes.get("Статус фации") != second.attributes.get("Статус фации"):
        return False
    first_interval = first.attributes.get("__structural_interval")
    second_interval = second.attributes.get("__structural_interval")
    if (first_interval or second_interval) and first_interval != second_interval:
        return False
    one, two = QPolygonF(first.polygon).boundingRect(), QPolygonF(second.polygon).boundingRect()
    vertical_gap = two.top() - one.bottom()
    horizontal_overlap = max(0.0, min(one.right(), two.right()) - max(one.left(), two.left()))
    return -2.0 <= vertical_gap <= 8.0 and horizontal_overlap >= min(one.width(), two.width()) * 0.65


def _merged(first: FaciesDetection, second: FaciesDetection) -> FaciesDetection:
    one, two = QPolygonF(first.polygon).boundingRect(), QPolygonF(second.polygon).boundingRect()
    rect = one.united(two)
    return FaciesDetection(
        label=first.label,
        # Do not hide the least certain part of a merged band from review.
        confidence=min(first.confidence, second.confidence),
        polygon=[QPointF(rect.left(), rect.top()), QPointF(rect.right(), rect.top()), QPointF(rect.right(), rect.bottom()), QPointF(rect.left(), rect.bottom())],
        attributes=dict(first.attributes),
        depth_from=min(value for value in (first.depth_from, second.depth_from) if value is not None) if first.depth_from is not None or second.depth_from is not None else None,
        depth_to=max(value for value in (first.depth_to, second.depth_to) if value is not None) if first.depth_to is not None or second.depth_to is not None else None,
        alternatives={**first.alternatives, **second.alternatives},
    )
