"""Make a quarantined, core-disjoint visual pilot from USGS Midnight Pass.

Only one broad source label is transferred to each interior window. The two
held-out cores are never used by the training manifest. No facies, mineral or
grain-size targets are inferred from the photos.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/external_raw/usgs_midnight_pass"
OUT = ROOT / "data/usgs_midnight_pass_shell_pilot"
GEOMETRY = ROOT / "data/usgs_midnight_pass_interval_candidates/candidate_manifest.json"
PHOTOS = SOURCE / "2024-330-FA_MP_CorePhotos.zip"

# Explicit windows stay away from mapped source contacts. MP3-B/MP3-E are held
# out by core, including every window of those photos.
PLAN = {
    "train": {
        "MP2-E": [(4, 14, "SHELLY SAND"), (16, 26, "SHELLY SAND"),
                  (34, 44, "SHELLY SAND"), (42, 51, "SHELLY SAND")],
        "MP2-F": [(5, 15, "SHELL LAG"), (18, 28, "SHELL LAG"),
                  (31, 41, "SHELL LAG"), (44, 54, "SHELL LAG"),
                  (57, 67, "SHELL LAG")],
        "MP2-G": [(3, 14, "SHELL LAG"), (20, 30, "SHELL LAG"),
                  (34, 44, "SHELL LAG"), (56, 68, "SHELL LAG")],
        "MP3-D": [(55, 65, "SHELLY SAND"), (62, 72, "SHELLY SAND")],
    },
    "heldout": {
        "MP3-B": [(2, 9, "SHELL LAG"), (12.5, 19.5, "SHELLY SAND"),
                  (23, 27, "SHELL LAG"), (31, 39, "SHELL LAG"),
                  (43, 52, "SHELL LAG")],
        "MP3-E": [(4, 16, "SHELL LAG"), (24.5, 33.5, "SHELLY SAND"),
                  (96, 103, "SAND")],
    },
}

TARGET = {
    "SHELL LAG": "На поверхности виден раковинный материал: многочисленные фрагменты раковин с песчаным материалом между ними.",
    "SHELLY SAND": "На поверхности виден песчаный материал с многочисленными фрагментами раковин.",
    "SAND": "На поверхности преобладает светлый песчаный материал; встречаются отдельные раковинные фрагменты.",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image_dir = OUT / "images"
    image_dir.mkdir(exist_ok=True)
    source = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    intervals = source["interval_candidates"]
    train_ids = set(PLAN["train"])
    heldout_ids = set(PLAN["heldout"])
    if train_ids & heldout_ids:
        raise ValueError("Core ID leakage across train and heldout")
    records = {"train": [], "heldout": []}
    thumbs = []
    with zipfile.ZipFile(PHOTOS) as archive:
        for split, cores in PLAN.items():
            for core_id, windows in cores.items():
                with Image.open(io.BytesIO(archive.read(f"{core_id}.jpg"))) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                fit = source["cores"][core_id]
                scale = image.width / fit["preview_width_px"]
                for number, (top, base, label) in enumerate(windows, 1):
                    matching = [r for r in intervals if r["core_id"] == core_id
                                and r["source_lithology_en"] == label
                                and r["source_interval_cm"][0] + 1.5 <= top
                                and base <= r["source_interval_cm"][1] - 1.5]
                    if len(matching) != 1:
                        raise ValueError(f"Window lacks unique source interval: {core_id} {top}-{base} {label}")
                    x0 = fit["x_at_zero_cm_preview"]
                    step = fit["pixels_per_cm_preview"]
                    box = (round((x0 + top * step) * scale), round(image.height * .025),
                           round((x0 + base * step) * scale), round(image.height * .69))
                    if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                        raise ValueError(f"Bad crop: {core_id} {box}")
                    crop = image.crop(box)
                    filename = f"{split}_{core_id}_{number:02d}_{top:g}-{base:g}cm.jpg"
                    path = image_dir / filename
                    crop.save(path, quality=93, optimize=True)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    top_m, base_m = round(top / 100, 4), round(base / 100, 4)
                    prompt = (f"Это увеличенный фрагмент керна {core_id}, участок {top:g}–{base:g} см от верха керна. "
                              "Опиши по-русски только хорошо видимое соотношение песчаного и раковинного материала. "
                              "Не угадывай среду осадконакопления, минералы, размер зёрен и признаки вне кадра. "
                              "Верни только JSON: description_segments с одним объектом top_m, base_m, description.")
                    target = json.dumps({"description_segments": [{"top_m": top_m,
                                     "base_m": base_m, "description": TARGET[label]}]}, ensure_ascii=False)
                    record = {"image": str(path.resolve()),
                              "conversations": [{"from": "human", "value": "<image>\n" + prompt},
                                                {"from": "gpt", "value": target}],
                              "metadata": {"record_id": f"midnight_{split}_{core_id}_{number:02d}",
                                           "record_type": "source_labeled_visual_window_pilot",
                                           "source_group": "USGS_Midnight_Pass_2025",
                                           "core_id": core_id, "split": split,
                                           "source_url": source["source_url"],
                                           "license": source["license"],
                                           "source_log_member": matching[0]["source_log_member"],
                                           "source_photo_member": matching[0]["source_photo_member"],
                                           "source_interval_cm": matching[0]["source_interval_cm"],
                                           "window_cm": [top, base], "source_lithology_en": label,
                                           "photo_crop_box_px": list(box), "image_sha256": digest,
                                           "source_log_transcribed_manually": True,
                                           "ruler_calibration_visually_checked": True,
                                           "visual_target_reviewed_by_assistant": True,
                                           "geology_approved": False,
                                           "training_approved": False}}
                    records[split].append(record)
                    tile = crop.copy()
                    tile.thumbnail((300, 165))
                    thumbs.append((filename, label, tile))
    for split in records:
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for record in records[split]:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    width, cell_w, cell_h = 1500, 300, 230
    sheet = Image.new("RGB", (width, ((len(thumbs) + 4) // 5) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (filename, label, tile) in enumerate(thumbs):
        x, y = (i % 5) * cell_w, (i // 5) * cell_h
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + 170), filename[:37], fill="#152c42")
        draw.text((x + 4, y + 190), label, fill="#37536e")
    sheet.save(OUT / "contact_sheet.jpg", quality=90)
    summary = {"train_cores": sorted(train_ids), "heldout_cores": sorted(heldout_ids),
               "train_windows": len(records["train"]), "heldout_windows": len(records["heldout"]),
               "train_labels": dict(Counter(r["metadata"]["source_lithology_en"] for r in records["train"])),
               "heldout_labels": dict(Counter(r["metadata"]["source_lithology_en"] for r in records["heldout"])),
               "training_approved": 0, "purpose": "experimental visual transfer only; not product training"}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
