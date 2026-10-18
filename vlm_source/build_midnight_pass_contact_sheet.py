"""Generate a compact visual audit sheet for quarantined Midnight Pass crops."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent / "data/usgs_midnight_pass_interval_candidates"


def main() -> None:
    records = json.loads((ROOT / "candidate_manifest.json").read_text(encoding="utf-8"))["interval_candidates"]
    columns, tile_width, tile_height = 4, 400, 300
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    for index, row in enumerate(records):
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        with Image.open(row["crop_path"]) as source:
            picture = source.convert("RGB")
            picture.thumbnail((tile_width - 18, 226))
            sheet.paste(picture, (x + (tile_width - picture.width) // 2, y + 5))
        top, base = row["source_interval_cm"]
        draw.text((x + 10, y + 239), f"{row['core_id']}  {top:g}-{base:g} cm", fill="#172936", font=font)
        draw.text((x + 10, y + 264), row["source_lithology_en"], fill="#476170", font=font)
    output = ROOT / "contact_sheet.jpg"
    sheet.save(output, quality=90, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
