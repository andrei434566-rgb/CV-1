"""Build a Russian lithology pilot from Cat Island's depth-linked logs.

Only single-lithology spans of at least 0.28 m are cropped, leaving a margin
around contacts.  Split by entire core to avoid adjacent-image leakage.
The resulting rows remain experimental candidates, never approved geology.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "cat_island_candidates"
OUT = ROOT / "data" / "cat_island_lithology_pilot"
LITHOLOGY = {
    "SAND": "песок",
    "CLAY": "глина",
    "MUD": "ил",
    "MUDDY SAND": "глинистый песок",
    "SANDY MUD": "песчанистый ил",
    "SILTY MUD": "алевритистый ил",
    "SANDY CLAY": "песчанистая глина",
    "SILTY CLAY": "алевритистая глина",
    "PEAT": "торф",
}
PROMPT = (
    "<image>\nНазови преобладающую породу в показанном фрагменте керна. "
    "Ответь только JSON с ключом lithology на русском языке. "
    "Если по фото породу нельзя уверенно различить, укажи «неопределено»."
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images = OUT / "images"
    images.mkdir(exist_ok=True)
    source = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    cores = {row["core_id"]: row for row in source["cores"]}
    candidates = [json.loads(line) for line in (SOURCE / "source_candidates.jsonl").read_text(encoding="utf-8").splitlines() if line]
    groups: dict[str, list[dict]] = {"terrestrial_beach_ridge": [], "marine": []}
    for core in cores.values():
        groups[core["setting"]].append(core)
    heldout = set()
    for setting, group in groups.items():
        ordered = sorted(group, key=lambda core: hashlib.sha256(core["core_id"].encode()).hexdigest())
        heldout.update(core["core_id"] for core in ordered[:max(2, round(len(ordered) * 0.2))])

    train, holdout = [], []
    excluded = {"no_translation": 0, "short_interval": 0, "depth_gap": 0}
    for row in candidates:
        meta = row["metadata"]
        core = cores[meta["source_well"]]
        if core["description_gaps_m"]:
            excluded["depth_gap"] += 1
            continue
        photo_top, photo_base = meta["photo_interval_m"]
        with Image.open(row["image"]) as source_image:
            image = source_image.convert("RGB")
        for index, layer in enumerate(core["parsed_layers"]):
            lithology_en = layer["description"].split(",", 1)[0].strip().upper()
            lithology_ru = LITHOLOGY.get(lithology_en)
            if not lithology_ru:
                excluded["no_translation"] += 1
                continue
            top = max(photo_top, layer["top_m"]) + 0.04
            base = min(photo_base, layer["base_m"]) - 0.04
            if base - top < 0.20:
                excluded["short_interval"] += 1
                continue
            # Every photographed column has a 0–100 cm ruler, including the
            # final partial column.  Do not stretch a short last interval to
            # the full image height.
            y0 = round((top - photo_top) * image.height)
            y1 = round((base - photo_top) * image.height)
            crop = image.crop((0, max(0, y0), image.width, min(image.height, y1)))
            if crop.height < 100:
                excluded["short_interval"] += 1
                continue
            path = images / f"{meta['source_well']}_{meta['record_id'].split(':')[-1]}_layer{index:02d}.jpg"
            crop.save(path, quality=95)
            record = {
                "image": str(path),
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt", "value": json.dumps({"lithology": lithology_ru}, ensure_ascii=False)},
                ],
                "metadata": {
                    "record_id": path.stem, "dataset": "usgs_2010_cat_island",
                    "source_well": meta["source_well"], "source_photo": row["image"],
                    "source_pdf": meta["source_pdf"], "source_url": meta["source_url"],
                    "setting_context": meta["setting_context"], "source_interval_m": [round(top, 3), round(base, 3)],
                    "lithology_source_en": lithology_en, "lithology_target_ru": lithology_ru,
                    "source_description_en": layer["description"],
                    "training_approved": False,
                    "review_status": "experimental_automatic_crop_and_dictionary_translation",
                },
            }
            (holdout if meta["source_well"] in heldout else train).append(record)

    for name, rows in (("train", train), ("heldout", holdout)):
        (OUT / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
        )
    summary = {
        "purpose": "experimental Russian lithology-only adapter, not a full core-description model",
        "source_url": "https://pubs.usgs.gov/ds/0834/ds834_data.html",
        "train_images": len(train), "heldout_images": len(holdout),
        "train_cores": sorted({row["metadata"]["source_well"] for row in train}),
        "heldout_cores": sorted({row["metadata"]["source_well"] for row in holdout}),
        "reserved_heldout_cores": sorted(heldout),
        "train_lithologies": sorted({row["metadata"]["lithology_target_ru"] for row in train}),
        "excluded": excluded,
        "training_approved": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("train_images", "heldout_images", "heldout_cores", "excluded")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
