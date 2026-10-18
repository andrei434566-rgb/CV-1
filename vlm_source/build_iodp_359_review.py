"""Crop empty black canvas from IODP 359 source images for visual review only."""

from __future__ import annotations

import html
import io
import json
import statistics
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parent / "data" / "external_raw" / "iodp_359_photo_pilot"
CANDIDATES = ROOT / "lily_alignment_candidates.jsonl"
PREVIEWS = ROOT / "review_previews"


def main() -> None:
    PREVIEWS.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line]
    by_zip = {}
    cards = []
    widths = []
    for row in rows:
        archive_name = row["photo_zip"]
        if archive_name not in by_zip:
            by_zip[archive_name] = zipfile.ZipFile(ROOT / archive_name)
        with by_zip[archive_name].open(row["photo_zip_member"]) as stream:
            image = Image.open(io.BytesIO(stream.read())).convert("RGB")
        bbox = image.getbbox()
        if bbox is None:
            row["review_issue"] = "blank_image"
            continue
        widths.append(bbox[2] - bbox[0])
        pad = 15
        crop_box = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                    min(image.width, bbox[2] + pad), min(image.height, bbox[3] + pad))
        target = PREVIEWS / (row["record_id"] + ".jpg")
        if not target.exists():
            cropped = image.crop(crop_box)
            cropped.save(target, quality=93, optimize=True)
        row["nonblack_bbox_px"] = list(bbox)
        row["review_preview"] = target.relative_to(ROOT).as_posix()
        label = html.escape(row["dominant_lithology_en"] or "без метки")
        depth = row["photo_depth_csf_a_m"]
        members = "<br>".join(f"{m['overlap_top_m']:.2f}–{m['overlap_bottom_m']:.2f} м: {html.escape(m['full_lithology_en'])}" for m in row["matched_lithology_rows"])
        quality = "priority" if row["lithology_covered_fraction"] >= .95 and row["dominant_fraction_of_photo"] >= .9 else "review"
        cards.append(f"<details class='{quality}'><summary>{html.escape(row['record_id'])}: {depth[0]:.2f}–{depth[1]:.2f} м, {label} ({row['dominant_fraction_of_photo']:.0%}), покрытие {row['lithology_covered_fraction']:.0%}</summary><div class='inside'><a href='{html.escape(row['review_preview'])}'><img loading='lazy' src='{html.escape(row['review_preview'])}'></a><div><b>Исходные интервалы LILY</b><p>{members}</p><p>Исходное PNG: {html.escape(row['photo_zip_member'])}</p><p>Ширина керна в исходном PNG: {bbox[2]-bbox[0]} пикс.</p></div></div></details>")
    for archive in by_zip.values():
        archive.close()
    CANDIDATES.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary_path = ROOT / "alignment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review_preview_count"] = len(cards)
    summary["median_native_nonblack_core_width_px"] = statistics.median(widths) if widths else None
    summary["review_index"] = "review_index.html"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    document = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>IODP 359 + LILY: проверка пар</title><style>body{font:16px sans-serif;max-width:1200px;margin:2rem auto;color:#21323a;background:#f8fafb}details{background:white;border:1px solid #ccd;border-radius:.5rem;margin:.7rem 0;padding:.7rem}summary{cursor:pointer}.priority{border-left:5px solid #2b8a57}.inside{display:flex;gap:1.5rem;margin:1rem;align-items:flex-start}.inside img{max-height:600px;max-width:420px;image-rendering:auto}.inside div{overflow-wrap:anywhere}</style><h1>IODP Expedition 359 + LILY: проверка фото и литологии</h1><p>Фото: <a href='https://zenodo.org/records/14015049'>IODP 359</a>; интервалы: <a href='https://zenodo.org/records/10425539'>LILY v2</a>. Оба источника CC BY 4.0. Зелёным выделены кандидаты с покрытием ≥95% и преобладанием одной литологии ≥90%; это не означает, что подтип породы виден на фотографии. Чёрный фон убран только в предпросмотре; исходные PNG сохранены. Обучающих меток пока 0.</p>""" + "".join(cards) + "</html>"
    (ROOT / "review_index.html").write_text(document, encoding="utf-8")
    print(f"Previews: {len(cards)}; median native core width: {summary['median_native_nonblack_core_width_px']} px")


if __name__ == "__main__":
    main()
