"""Read the UKCS Figshare lithology arrays and summarize trainable intervals."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "external_raw" / "ukcs_q204_q205_coremdlr" / "Q204_data.zip"
OUTPUT = ROOT / "data" / "ukcs_q204_q205_inventory" / "label_summary.json"
LABELS = {
    "s": "песчаник",
    "is": "глинистый песчаник",
    "ih": "песчанистый аргиллит",
    "os": "нефтенасыщенный песчаник",
    "sh": "аргиллит",
}
EXCLUDE = {"nc": "нет керна", "bs": "некачественное изображение песчаника"}
WINDOW_PX = 600


def runs(values: np.ndarray, depth: np.ndarray) -> list[tuple[str, int, int]]:
    changes = (values[1:] != values[:-1]) | (np.diff(depth) < 0) | (np.diff(depth) > 0.01)
    bounds = np.r_[0, np.flatnonzero(changes) + 1, len(values)]
    return [(values[start].decode("ascii"), int(start), int(end)) for start, end in zip(bounds[:-1], bounds[1:])]


def main() -> None:
    wells = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        for name in sorted(n for n in names if n.endswith("_labels.npy")):
            well = Path(name).name.removesuffix("_labels.npy")
            prefix = name.removesuffix("_labels.npy")
            with archive.open(name) as member:
                labels = np.load(member, allow_pickle=False)
            with archive.open(prefix + "_depth.npy") as member:
                depth = np.load(member, allow_pickle=False)
            if labels.ndim != 1 or depth.ndim != 1 or len(labels) != len(depth):
                raise ValueError(f"Incompatible depth/label vectors: {well}")
            if prefix + "_image.npy" not in names:
                raise ValueError(f"Missing image array: {well}")
            pieces = runs(labels, depth)
            counts = Counter(k.decode("ascii") for k in labels)
            eligible = Counter()
            for code, start, end in pieces:
                if code in LABELS:
                    eligible[code] += (end - start) // WINDOW_PX
            wells.append({
                "well": well,
                "rows": len(labels),
                "depth_first": float(depth[0]),
                "depth_last": float(depth[-1]),
                "depth_monotonic_increasing": bool(np.all(np.diff(depth) >= 0)),
                "depth_breaks": int(np.sum((np.diff(depth) < 0) | (np.diff(depth) > 0.01))),
                "pixel_counts": dict(counts),
                "contiguous_runs": dict(Counter(code for code, _, _ in pieces)),
                "pure_nonoverlapping_600px_windows": dict(eligible),
                "image_member": prefix + "_image.npy",
            })
    report = {
        "source": "https://figshare.com/articles/dataset/UKCS_Q204_Q205_Subsurface_Data_products_for_Machine_Learning_Study/14265689",
        "well_count": len(wells),
        "window_height_px": WINDOW_PX,
        "classes_ru": LABELS,
        "excluded_codes": EXCLUDE,
        "wells": wells,
        "all_well_candidate_windows": dict(sum((Counter(w["pure_nonoverlapping_600px_windows"]) for w in wells), Counter())),
        "status": "candidate counts only; window selection and photo content require geological review; final train/holdout split must be by entire well",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("well_count", "all_well_candidate_windows", "wells")}, ensure_ascii=True, indent=2)[:15000])


if __name__ == "__main__":
    main()
