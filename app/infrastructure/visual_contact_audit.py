"""Review-only audit of curved visual contacts between adjacent draft intervals.

This module does not classify facies, estimate a calibrated probability,
or approve polygons for training. It operates on predictions already produced
by create_automatic_interval_detections and preserves their source coordinates.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


def is_uv_photo(path: str | Path) -> bool:
    """Exclude UV/ultraviolet captures from the ordinary visible-light workflow."""
    stem = Path(path).stem.casefold()
    return bool(re.search(r"(?<![a-zа-яё0-9])uv(?![a-zа-яё0-9])|ультрафиолет", stem))


def _xy(point) -> tuple[float, float]:
    if hasattr(point, "x") and callable(point.x):
        return float(point.x()), float(point.y())
    return float(point[0]), float(point[1])


def _top_trace(points) -> list[list[float]]:
    """Read the top contact from existing curved interval polygons.

    The existing package-polygon builder walks left-to-right along its
    upper contact and returns along the lower contact. For a rectangular
    fallback polygon, the first two points are its top edge.
    """
    pts = [_xy(p) for p in points]
    if len(pts) < 4:
        return []
    upper = pts[: len(pts) // 2]
    if not upper:
        return []
    unique = {}
    for x, y in upper:
        unique[round(x, 3)] = [round(x, 2), round(y, 2)]
    return list(sorted(unique.values(), key=lambda item: item[0]))


def _compact_trace(points: list[list[float]], maximum: int = 80) -> list[list[float]]:
    if len(points) <= maximum:
        return points
    indexes = [round(i * (len(points) - 1) / (maximum - 1)) for i in range(maximum)]
    return [points[i] for i in indexes]


def audit_visual_contacts(detections, *, source_photo: str = "") -> dict:
    """Summarize adjacent visual packages from the same column, with depth if known.

    Contacts are hypotheses, not facies labels; empty results are legitimate.
    """
    by_column: dict[int, list] = defaultdict(list)
    for d in detections:
        attributes = getattr(d, "attributes", {}) or {}
        match = re.fullmatch(r"(\d+):(\d+)", str(attributes.get("__structural_interval", "")))
        if match and len(getattr(d, "polygon", [])) >= 4:
            by_column[int(match.group(1))].append((int(match.group(2)), d))

    intervals, contacts = [], []
    for column_index in sorted(by_column):
        ordered = sorted(by_column[column_index], key=lambda pair: pair[0])
        for position, d in ordered:
            pts = [_xy(p) for p in d.polygon]
            intervals.append({
                "column": column_index, "index": position,
                "depth_from_m": getattr(d, "depth_from", None),
                "depth_to_m": getattr(d, "depth_to", None),
                "y_from_px": round(min(y for _, y in pts), 2),
                "y_to_px": round(max(y for _, y in pts), 2),
                "reviewed": False,
            })
        for (upper_index, upper), (lower_index, lower) in zip(ordered[:-1], ordered[1:]):
            if lower_index != upper_index + 1:
                continue
            trace = _top_trace(lower.polygon)
            if len(trace) < 2:
                continue
            top_depth = getattr(upper, "depth_to", None)
            bottom_depth = getattr(lower, "depth_from", None)
            depth = None
            depth_conflict = False
            if top_depth is not None and bottom_depth is not None:
                depth = round((float(top_depth) + float(bottom_depth)) / 2, 3)
                depth_conflict = abs(float(top_depth) - float(bottom_depth)) > 0.02
            ys = [p[1] for p in trace]
            amplitude = round(max(ys) - min(ys), 2)
            contacts.append({
                "column": column_index,
                "above_interval_index": upper_index,
                "below_interval_index": lower_index,
                "depth_m": depth,
                "pixel_y": round(sum(ys) / len(ys), 2),
                "amplitude_px": amplitude,
                "geometry": "curved" if amplitude >= 3.0 else "near_horizontal",
                "trace": _compact_trace(trace),
                "depth_conflict": depth_conflict,
                "source": "visual_texture_heuristic",
                "geologist_reviewed": False,
                "training_approved": False,
            })
    return {
        "schema": "cv1-visual-contacts-review-v1",
        "source_photo": source_photo,
        "column_count": len(by_column),
        "interval_count": len(intervals),
        "contact_count": len(contacts),
        "intervals": intervals,
        "contacts": contacts,
        "limitations": [
            "Texture boundaries are suggestions, not verified geological facies contacts.",
            "Curves follow image contrast and can be displaced by cracks, sample labels and lighting.",
            "Depths are only estimates from the current photo/column calibration.",
            "Reviewing this export does not approve labels or polygons for training.",
        ],
    }
