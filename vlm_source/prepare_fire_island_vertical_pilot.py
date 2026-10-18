"""Make ruler-aligned *review candidates* from two vertical USGS core photos.

The 10-cm tick pixel positions below were inspected on the original photos.
No crop or Russian draft in this file is automatically admitted to training.
"""

from __future__ import annotations

import hashlib
import html
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_fire_island_2016"
OUT = Path(__file__).resolve().parent / "data/usgs_fire_island_vertical_pilot"
CALIBRATION = {
    "P22-C14": {
        "tick_y_px": {0: 144, 10: 493, 20: 841, 30: 1192, 40: 1548,
                      50: 1913, 60: 2277, 70: 2642, 80: 3007, 90: 3373, 100: 3739},
        "core_x_px": [552, 837],
        "ruler_and_core_x_px": [450, 845],
    },
    "P25-C05": {
        "tick_y_px": {0: 313, 10: 653, 20: 1000, 30: 1354, 40: 1713,
                      50: 2077, 60: 2434, 70: 2783, 80: 3123, 90: 3453, 100: 3789},
        "core_x_px": [840, 1090],
        "ruler_and_core_x_px": [825, 1190],
    },
}
# Deliberately away from source interval boundaries and (except for 14-28 cm)
# limited to visual features seen on the full photo. Targets are review drafts.
WINDOWS = [
    ("P22-C14", 3, 10, "Светлый преимущественно однородный песок."),
    ("P22-C14", 14, 28, "Светлый песок с двумя тёмными вытянутыми изогнутыми участками."),
    ("P22-C14", 48, 70, "Светлый песок с тёмными деформированными прослоями и линзами."),
    ("P22-C14", 85, 96, "Светлый песок с наклонным тёмным прослоем."),
    ("P25-C05", 10, 35, "Светлый песок без отчётливой слоистости."),
    ("P25-C05", 56, 68, "Светлый песок с тёмными изогнутыми прослоями."),
    ("P25-C05", 75, 85, "Светлый песок с тёмными деформированными прослоями."),
]


def depth_to_y(core_id: str, depth_cm: float) -> int:
    ticks = CALIBRATION[core_id]["tick_y_px"]
    lo = int(depth_cm // 10) * 10
    if lo == 100:
        return ticks[100]
    hi = lo + 10
    if lo not in ticks or hi not in ticks:
        raise ValueError(f"Depth outside calibrated first column: {core_id} {depth_cm}")
    return round(ticks[lo] + (depth_cm - lo) / 10 * (ticks[hi] - ticks[lo]))


def main() -> None:
    source = json.loads((ROOT / "source_intervals.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    records = []
    with ZipFile(ROOT / "2016-322-FA_CorePhotos.zip") as archive:
        for core_id, top, base, draft in WINDOWS:
            parents = [r for r in source["intervals"]
                       if r["core_id"] == core_id
                       and r["source_interval_cm"][0] <= top
                       and base <= r["source_interval_cm"][1]]
            if len(parents) != 1:
                raise ValueError(f"Expected one source interval for {core_id} {top}-{base}")
            parent = parents[0]
            member = f"2016-322-FA_{core_id}.jpg"
            raw = archive.read(member)
            image = Image.open(BytesIO(raw)).convert("RGB")
            calibration = CALIBRATION[core_id]
            y0, y1 = depth_to_y(core_id, top), depth_to_y(core_id, base)
            x0, x1 = calibration["core_x_px"]
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ValueError(f"Crop outside image: {core_id} {top}-{base}")
            stem = f"{core_id}_{top:g}-{base:g}cm"
            core_path = OUT / "images" / f"{stem}_core.jpg"
            ruler_path = OUT / "images" / f"{stem}_ruler.jpg"
            image.crop((x0, y0, x1, y1)).save(core_path, quality=95)
            r0, r1 = calibration["ruler_and_core_x_px"]
            image.crop((r0, y0, r1, y1)).save(ruler_path, quality=92)
            records.append({
                "id": f"fire_island_2016_{stem}", "core_id": core_id,
                "crop_interval_cm": [top, base],
                "source_interval_cm": parent["source_interval_cm"],
                "source_record_id": parent["record_id"],
                "source_pdf_page": parent["source_pdf_page"],
                "source_description_en": parent["source_description_en"],
                "visible_target_ru_draft": draft,
                "photo_member": member,
                "photo_member_sha256": hashlib.sha256(raw).hexdigest(),
                "crop_box_xyxy_px": [x0, y0, x1, y1],
                "image_size_px": list(image.size),
                "core_crop": str(core_path.relative_to(OUT)).replace("\\", "/"),
                "core_crop_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
                "ruler_crop": str(ruler_path.relative_to(OUT)).replace("\\", "/"),
                "alignment_method": "manual 10-cm labelled ruler ticks, linear interpolation inside each 10-cm span",
                "photo_interval_aligned": True,
                "russian_visual_target_reviewed": False,
                "training_approved": False,
            })
    manifest = {
        "source_url": source["source_url"],
        "source_pdf_sha256": source["source_pdf_sha256"],
        "license": "CC0 1.0 Universal",
        "ruler_calibration": CALIBRATION,
        "alignment_scope": "first photographed 0-100 cm column of P22-C14 and P25-C05 only",
        "candidates": records,
        "candidate_count": len(records), "training_approved_count": 0,
        "warning": "Russian visible targets are drafts, not geologist-reviewed; do not use in production training.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for record in records:
        cards.append(
            "<article data-id='" + html.escape(record["id"]) + "'><h2>" + html.escape(record["core_id"]) + " · "
            + f"{record['crop_interval_cm'][0]:g}–{record['crop_interval_cm'][1]:g} см"
            + "</h2><div class='images'><img src='" + html.escape(record["core_crop"])
            + "'><img src='" + html.escape(record["ruler_crop"])
            + "'></div><p><b>Черновик визуальной цели:</b> "
            + html.escape(record["visible_target_ru_draft"])
            + "</p><p><b>Оригинальное описание интервала:</b> "
            + html.escape(record["source_description_en"])
            + "</p><p><label>Исправленное визуальное описание (только видимое на вырезке):<br>"
            + "<textarea class='target' rows='3'>" + html.escape(record["visible_target_ru_draft"])
            + "</textarea></label></p><p><label><input class='approved' type='checkbox'>"
            + " Подтверждаю привязку к глубине и исправленное описание как экспертную метку</label></p>"
            + "<p>Фото: " + html.escape(record["photo_member"])
            + f"; PDF: стр. {record['source_pdf_page']}; пиксели: "
            + html.escape(str(record["crop_box_xyxy_px"])) + "</p></article>"
        )
    html_page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>Fire Island: вертикальные окна</title>
<style>body{font:16px system-ui;margin:2rem auto;max-width:1400px;background:#f5f7fa;color:#223}article{background:white;margin:1.5rem 0;padding:1.5rem;border:1px solid #ccd;border-radius:12px}.images{display:flex;gap:1rem;align-items:flex-start;overflow:auto}.images img{max-width:46%;max-height:800px;object-fit:contain}p{line-height:1.5}textarea{box-sizing:border-box;width:100%;font:inherit;padding:.5rem}button,input{font:inherit}button{padding:.6rem 1rem}</style>
<h1>USGS Fire Island: 7 окон, привязанных к линейке</h1><p>Слева вырезка керна, справа вырезка вместе с линейкой. Русские формулировки — черновики только по видимым признакам; 0 примеров допущено к обучению без проверки геологом. Исправления сохраняются в этом браузере. Отметьте галочками только точные пары, затем выгрузите JSON для включения в учебный корпус.</p>
<p><label>Имя проверяющего: <input id='reviewer' size='30'></label> <button id='export' type='button'>Скачать результаты проверки</button></p>""" + "".join(cards) + """
<script type='application/json' id='manifest'>""" + json.dumps({
        "source_pdf_sha256": source["source_pdf_sha256"],
        "records": [{"id": r["id"], "core_crop_sha256": r["core_crop_sha256"]} for r in records],
    }, ensure_ascii=False).replace("<", "\\u003c") + """</script>
<script>
const manifest=JSON.parse(document.getElementById('manifest').textContent);
const key='fire-island-2016-vertical-pilot-review-v1';
const cards=[...document.querySelectorAll('article[data-id]')];
const saved=JSON.parse(localStorage.getItem(key)||'{}');
document.getElementById('reviewer').value=saved.reviewer||'';
for(const card of cards){const r=saved.rows?.[card.dataset.id];if(r){card.querySelector('.target').value=r.text;card.querySelector('.approved').checked=!!r.approved;}}
function state(){const rows={};for(const card of cards){rows[card.dataset.id]={text:card.querySelector('.target').value.trim(),approved:card.querySelector('.approved').checked};}return {reviewer:document.getElementById('reviewer').value.trim(),rows};}
document.addEventListener('input',()=>localStorage.setItem(key,JSON.stringify(state())));
document.addEventListener('change',()=>localStorage.setItem(key,JSON.stringify(state())));
document.getElementById('export').addEventListener('click',()=>{
 const s=state();if(Object.values(s.rows).some(r=>r.approved)&&!s.reviewer){alert('Укажите имя проверяющего для подтверждённых записей.');return;}
 const records=manifest.records.map(r=>({id:r.id,core_crop_sha256:r.core_crop_sha256,approved:s.rows[r.id].approved,description_ru:s.rows[r.id].text}));
 const result={source_pdf_sha256:manifest.source_pdf_sha256,reviewer:s.reviewer,reviewed_at:new Date().toISOString(),records};
 const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json;charset=utf-8'});
 const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='fire_island_vertical_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
});
</script></html>"""
    (OUT / "review.html").write_text(html_page, encoding="utf-8")
    print(json.dumps({"candidates": len(records), "cores": len({r['core_id'] for r in records}),
                      "training_approved": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
