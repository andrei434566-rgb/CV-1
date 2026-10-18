"""Stage depth-aligned UKCS photo crops with five provisional Russian labels.

This does not train a model. All emitted rows remain unapproved until a
geologist checks photo content, depth and the translated lithology label.
"""

from __future__ import annotations

import csv
import html
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_ukcs_labels import ARCHIVE, LABELS, ROOT, WINDOW_PX, runs


OUTPUT = ROOT / "data" / "ukcs_q204_q205_lithology_candidates"
MAX_PER_CLASS_PER_WELL = 8
PROMPT = (
    "Определи основную литологию на фрагменте фото керна. "
    "Выбери ровно один вариант: песчаник, глинистый песчаник, "
    "песчанистый аргиллит, нефтенасыщенный песчаник, аргиллит. "
    "Ответь только названием породы на русском языке."
)


def select_windows(labels: np.ndarray, depth: np.ndarray) -> list[tuple[int, int, str, int, int]]:
    candidates: dict[str, list[tuple[int, int, str, int, int]]] = defaultdict(list)
    for code, start, end in runs(labels, depth):
        if code not in LABELS or end - start < WINDOW_PX:
            continue
        count = (end - start) // WINDOW_PX
        for index in sorted({count // 2, max(0, count // 2 - 1)}):
            top = start + index * WINDOW_PX
            candidates[code].append((top, top + WINDOW_PX, code, start, end))
    selected = []
    for code, windows in candidates.items():
        if len(windows) > MAX_PER_CLASS_PER_WELL:
            indices = np.linspace(0, len(windows) - 1, MAX_PER_CLASS_PER_WELL).round().astype(int)
            windows = [windows[index] for index in sorted(set(indices))]
        selected.extend(windows)
    return sorted(selected)


def skip(stream, byte_count: int) -> None:
    while byte_count:
        block = stream.read(min(byte_count, 4 * 1024 * 1024))
        if not block:
            raise EOFError("Image ended before selected crop")
        byte_count -= len(block)


def read_crops(archive: zipfile.ZipFile, image_member: str, windows: list[tuple[int, int, str, int, int]]) -> list[np.ndarray]:
    result = []
    with archive.open(image_member) as stream:
        version = np.lib.format.read_magic(stream)
        reader = {1: np.lib.format.read_array_header_1_0, 2: np.lib.format.read_array_header_2_0}[version[0]]
        shape, fortran, dtype = reader(stream)
        if fortran or len(shape) != 3 or shape[2] != 3 or dtype != np.dtype("uint8"):
            raise ValueError(f"Unexpected image layout: {image_member} {shape} {dtype}")
        row_bytes = shape[1] * 3
        row = 0
        for top, bottom, _, _, _ in windows:
            if top < row:
                raise ValueError("Overlapping selected windows")
            skip(stream, (top - row) * row_bytes)
            image = np.frombuffer(stream.read((bottom - top) * row_bytes), dtype=np.uint8)
            if image.size != (bottom - top) * row_bytes:
                raise EOFError(f"Truncated image crop in {image_member}")
            result.append(image.reshape(bottom - top, shape[1], 3).copy())
            row = bottom
    return result


def trim_black_edges(image: np.ndarray) -> np.ndarray:
    bright = np.max(image, axis=2) > 12
    columns = np.flatnonzero(bright.mean(axis=0) > 0.25)
    if len(columns) < image.shape[1] // 3:
        return image
    left = max(0, int(columns[0]) - 12)
    right = min(image.shape[1], int(columns[-1]) + 13)
    return image[:, left:right]


def main() -> None:
    crop_dir = OUTPUT / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = set(archive.namelist())
        for label_member in sorted(name for name in members if name.endswith("_labels.npy")):
            prefix = label_member.removesuffix("_labels.npy")
            well = Path(prefix).name
            with archive.open(label_member) as stream:
                labels = np.load(stream, allow_pickle=False)
            with archive.open(prefix + "_depth.npy") as stream:
                depth = np.load(stream, allow_pickle=False)
            if len(labels) != len(depth):
                raise ValueError(f"Depth/label mismatch: {well}")
            windows = select_windows(labels, depth)
            image_member = prefix + "_image.npy"
            if image_member not in members:
                raise ValueError(f"Missing image member: {well}")
            crops = read_crops(archive, image_member, windows)
            for index, ((top, bottom, code, run_start, run_end), rgb) in enumerate(zip(windows, crops), 1):
                safe_well = well.replace("/", "_")
                record_id = f"{safe_well}_{index:03d}_{code}"
                output_image = crop_dir / f"{record_id}.jpg"
                Image.fromarray(trim_black_edges(rgb), "RGB").save(output_image, quality=91)
                record = {
                    "image": str(output_image),
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + PROMPT},
                        {"from": "gpt", "value": LABELS[code]},
                    ],
                    "metadata": {
                        "record_id": record_id,
                        "source_id": "ukcs_q204_q205_coremdlr",
                        "source_url": "https://figshare.com/articles/dataset/UKCS_Q204_Q205_Subsurface_Data_products_for_Machine_Learning_Study/14265689",
                        "well_id": well,
                        "image_member": image_member,
                        "label_member": label_member,
                        "row_top": top,
                        "row_bottom": bottom,
                        "depth_top_m": round(float(depth[top]), 4),
                        "depth_bottom_m": round(float(depth[bottom - 1]), 4),
                        "source_label_code": code,
                        "source_run_id": f"{well}_{run_start}_{run_end}",
                        "translated_label_ru": LABELS[code],
                        "crop_reviewed_by_geologist": False,
                        "training_approved": False,
                    },
                }
                records.append(record)
            print(f"{well}: {len(windows)} candidates", flush=True)
    manifest = OUTPUT / "candidates.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "source": "https://figshare.com/articles/dataset/UKCS_Q204_Q205_Subsurface_Data_products_for_Machine_Learning_Study/14265689",
        "wells": len({r["metadata"]["well_id"] for r in records}),
        "candidates": len(records),
        "by_class": dict(Counter(r["metadata"]["source_label_code"] for r in records)),
        "by_well": dict(Counter(r["metadata"]["well_id"] for r in records)),
        "max_per_class_per_well": MAX_PER_CLASS_PER_WELL,
        "crop_height_px": WINDOW_PX,
        "geologist_approved": 0,
        "training_status": "staged_candidates_only; not in any model",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "well_id", "depth_top_m", "depth_bottom_m", "source_label_code", "translated_label_ru", "photo_ok", "depth_ok", "corrected_label_ru", "notes"])
        writer.writeheader()
        for row in records:
            meta = row["metadata"]
            writer.writerow({key: meta.get(key, "") for key in writer.fieldnames})
    cards = []
    for row in records:
        meta = row["metadata"]
        src = "crops/" + Path(row["image"]).name
        cards.append(
            f'<article><img loading="lazy" src="{html.escape(src)}"><p><b>{html.escape(meta["well_id"])}</b> '
            f'{meta["depth_top_m"]}–{meta["depth_bottom_m"]} м; '
            f'{html.escape(meta["translated_label_ru"])} ({meta["source_label_code"]})</p></article>'
        )
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>UKCS — проверка фрагментов</title>'
        '<style>body{font:16px system-ui;background:#eef1f3;color:#182026;margin:24px}'
        'main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}'
        'article{background:white;padding:12px;border-radius:8px}img{display:block;max-width:100%;height:300px;object-fit:contain}'
        'p{margin:8px 0 0}</style><h1>UKCS: кандидаты для геологической проверки</h1>'
        '<p>Автоматические метки и глубины. Для правок используйте review.csv. Ничего из этой страницы ещё не обучало модель.</p>'
        '<p>Contains British Geological Survey materials © UKRI 2019 &amp; 2020. '
        'Геологическая разметка: Thomas Martin; обработка данных: Ross Meyer.</p>'
        '<main>' + "".join(cards) + '</main></html>'
    )
    (OUTPUT / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
