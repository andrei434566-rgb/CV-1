"""Stage four visually inspected VC4 features for geologist review only.

Depth-to-pixel conversion uses whole-page photo endpoints and remains an
approximation; source descriptions are preserved, not used as training labels.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/external_raw/usgs_conconully_2025"
OUT = ROOT / "data/usgs_conconully_vc4_crop_candidates"
FEATURES = [
    (2, 60, 65, "conconully_vc4_03", "Тёмный серо-зелёный материал с неровной поверхностью, трещинами и почти чёрными участками."),
    (2, 75, 80, "conconully_vc4_04", "Неравномерно буровато-серый материал с трещинами и тёмными прожилками."),
    (5, 242, 252, "conconully_vc4_29", "Тёмные почти чёрные полосы чередуются со светло-серым материалом; границы неровные."),
    (6, 317, 323, "conconully_vc4_40", "В нижней части заметна тёмная зернистая зона со светлыми обломками."),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    page_index = json.loads((SOURCE / "page_index.json").read_text(encoding="utf-8"))
    descriptions = json.loads((SOURCE / "source_descriptions.json").read_text(encoding="utf-8"))
    by_id = {record["record_id"]: record for record in descriptions["records"]}
    by_page = {page["pdf_page"]: page for page in page_index["pages"]}
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for page_number, top, base, source_id, draft in FEATURES:
        page = by_page[page_number]
        source = by_id[source_id]
        segment_top, segment_base = page["segment_cm"]
        if not segment_top <= top < base <= segment_base:
            raise ValueError("Crop outside official page segment")
        original = SOURCE / page["photo_only_file"]
        if sha256(original) != page["photo_only_sha256"]:
            raise ValueError("Photo-only extract changed")
        with Image.open(original) as image:
            y0 = round((top - segment_top) / (segment_base - segment_top) * image.height)
            y1 = round((base - segment_top) / (segment_base - segment_top) * image.height)
            crop = image.crop((0, y0, image.width, y1)).convert("RGB")
            path = OUT / "images" / f"VC4_{top:03d}_{base:03d}.jpg"
            crop.save(path, quality=95, subsampling=0)
        records.append({
            "core_id": "VC4", "pdf_page": page_number,
            "crop_depth_window_cm_nominal": [top, base],
            "source_record_id": source_id,
            "source_record_depth_cm": source["source_depth_cm"],
            "source_description_en": source["source_description_en"],
            "visual_target_ru_draft": draft,
            "photo_only_sha256": page["photo_only_sha256"],
            "source_pdf_sha256": page_index["source_pdf_sha256"],
            "crop_box_xyxy_photo_px": [0, y0, page["photo_only_size_px"][0], y1],
            "crop_file": str(path.relative_to(OUT)).replace("\\", "/"),
            "crop_sha256": sha256(path),
            "alignment_method": "linear interpolation between page segment endpoints; precise ruler ticks not individually calibrated",
            "interval_pixel_aligned": False,
            "russian_visual_target_reviewed": False,
            "training_approved": False,
        })
    result = {"source_url": descriptions["source_url"], "doi": descriptions["doi"],
              "core_id": "VC4", "candidates": records,
              "summary": {"candidate_windows": len(records), "cores": 1, "approved_training_pairs": 0},
              "warning": "These are assistant-drafted visual observations on approximate crops from one core, not production labels or a held-out evaluation set."}
    (OUT / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for record in records:
        a, b = record["crop_depth_window_cm_nominal"]
        cards.append(f"<article><h2>VC4 · примерно {a}–{b} см</h2>"
                     f"<img src='{html.escape(record['crop_file'])}'>"
                     f"<p><b>Русский черновик по фото:</b> {html.escape(record['visual_target_ru_draft'])}</p>"
                     f"<p><b>Исходная запись {html.escape(record['source_record_id'])}:</b> "
                     f"{html.escape(record['source_description_en'])}</p>"
                     "<p>Точная пиксельная привязка и геологическая проверка отсутствуют. В обучении не используется.</p></article>")
    document = ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>Conconully VC4: кандидаты</title>"
                "<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;background:#f5f7fa;color:#223}"
                "article{background:white;border:1px solid #ccd;padding:1.2rem;margin:1.2rem 0}img{max-width:100%;max-height:600px}</style>"
                "<h1>USGS Conconully VC4: отдельные видимые признаки</h1>"
                "<p>Четыре кандидата из одного керна. Глубины по линейной шкале страниц приблизительны; "
                "русский текст — черновик, 0 пар допущено к обучению.</p>"
                + "".join(cards) + "</html>")
    (OUT / "review.html").write_text(document, encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
