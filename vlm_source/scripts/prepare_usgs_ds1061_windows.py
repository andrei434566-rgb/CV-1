"""Create weakly aligned lithology windows for human review, never training.

The printed remark is anchored at a page Y coordinate. Mapping its center to
the embedded photo is approximate because the remark is not an interval
boundary. Every output remains training_approved=False.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import pdfplumber
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "external_raw" / "usgs_cache_creek_ds1061" / "ds1061.pdf"
SECTION = ROOT / "data" / "usgs_cache_creek_ds1061_review"
OUT = ROOT / "data" / "usgs_cache_creek_ds1061_windows"
LABELS = {
    "Sand", "Silty sand", "Clayey sand", "Silt", "Sandy silt", "Clayey silt",
    "Clay", "Silty clay", "Sandy clay", "Sand-silt-clay",
    "Gravelly sand", "Gravelly silty sand", "Gravelly clayey sand",
    "Gravelly sandy silt", "Gravelly clayey silt", "Gravelly sandy clay",
    "Gravelly silty clay",
}


def main() -> None:
    source = json.loads((SECTION / "manifest.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    image_dir = OUT / "images"
    image_dir.mkdir(exist_ok=True)
    proposals = []
    with pdfplumber.open(PDF) as document:
        for record in source["records"]:
            page_number = record["pdf_page"]
            page = document.pages[page_number - 1]
            photos = [
                image for image in page.images
                if 15 <= image["x1"] - image["x0"] <= 80
                and image["bottom"] - image["top"] > 100
            ]
            if not photos:
                continue
            photo_rect = max(photos, key=lambda image: image["bottom"] - image["top"])
            words = page.extract_words()
            lines: dict[float, list[dict]] = {}
            for word in words:
                if 390 <= word["x0"] <= 475 and photo_rect["top"] <= word["top"] <= photo_rect["bottom"]:
                    lines.setdefault(round(word["top"], 1), []).append(word)
            image_path = ROOT / record["image"]
            with Image.open(image_path) as photo:
                for line_y, line_words in sorted(lines.items()):
                    line_words.sort(key=lambda word: word["x0"])
                    label = " ".join(word["text"] for word in line_words)
                    if label not in LABELS:
                        continue
                    fraction = (line_y - photo_rect["top"]) / (photo_rect["bottom"] - photo_rect["top"])
                    cy = round(fraction * photo.height)
                    half = 105
                    y0 = max(0, cy - half)
                    y1 = min(photo.height, cy + half)
                    if y1 - y0 < 120:
                        continue
                    stem = f"p{page_number:03d}_{len(proposals):04d}"
                    target = image_dir / f"{stem}.png"
                    photo.crop((0, y0, photo.width, y1)).save(target, optimize=True)
                    proposals.append({
                        "page": page_number,
                        "core_id": record["core_id"],
                        "borehole_id": record["borehole_id"],
                        "section": record["section"],
                        "source_depth_text": record["source_depth_text"],
                        "source_label_en": label,
                        "source_label_pdf_y_pt": line_y,
                        "image": str(target.relative_to(ROOT)).replace("\\", "/"),
                        "source_full_photo": record["image"],
                        "crop_y_px": [y0, y1],
                        "alignment": "approximate center of printed remark; NOT interval boundary",
                        "visual_label_reviewed": False,
                        "geology_approved": False,
                        "training_approved": False,
                    })
    counts = Counter(item["source_label_en"] for item in proposals)
    summary = {
        "source": "https://pubs.usgs.gov/ds/1061/ds1061.pdf",
        "candidate_windows": len(proposals),
        "source_pages": len({item["page"] for item in proposals}),
        "distinct_cores": len({item["core_id"] for item in proposals}),
        "distinct_boreholes": len({item["borehole_id"] for item in proposals}),
        "source_label_counts": dict(counts.most_common()),
        "training_approved": False,
        "reason": "Printed remark location is a point annotation, not an interval extent or verified visual label.",
    }
    (OUT / "manifest.json").write_text(json.dumps({"summary": summary, "proposals": proposals}, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for item in proposals:
        rel = "images/" + Path(item["image"]).name
        cards.append(f"<article><h3>{html.escape(item['core_id'])} · PDF p. {item['page']} · {html.escape(item['source_label_en'])}</h3>"
                     f"<img src='{html.escape(rel)}' alt='Предложенный фрагмент'>"
                     f"<p>{html.escape(item['source_depth_text'])}</p></article>")
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS DS1061 · кандидаты окон</title>
<style>body{font:15px system-ui;background:#f3f1ea;padding:1rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem}article{background:white;padding:1rem;border-radius:8px}img{max-width:100%;height:240px;object-fit:contain;background:#222}h3{font-size:1rem}</style>
<h1>Предложенные фрагменты по положениям подписей</h1><p>Ни один фрагмент не допущен в обучение. Подпись может относиться к соседнему или более длинному интервалу.</p><main>""" + "\n".join(cards) + "</main></html>"
    (OUT / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
