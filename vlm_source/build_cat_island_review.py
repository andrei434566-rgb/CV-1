"""Create an offline geology-review page for the Cat Island training crops."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cat_island_lithology_pilot"
OUTPUT = DATA / "review_train.html"


def main() -> None:
    rows = [json.loads(line) for line in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    labels = sorted({row["metadata"]["lithology_target_ru"] for row in rows} | {"неопределено"})
    cards = []
    for row in rows:
        meta = row["metadata"]
        image = Path(row["image"])
        src = "images/" + image.name
        choices = "".join(f'<option value="{html.escape(label)}"{(" selected" if label == meta["lithology_target_ru"] else "")}>{html.escape(label)}</option>' for label in labels)
        original = meta.get("source_description_en", meta["lithology_source_en"])
        cards.append(f'''
        <article class="card" data-record="{html.escape(meta['record_id'])}">
          <img src="{html.escape(src)}" alt="Керн {html.escape(meta['source_well'])}">
          <div class="body">
            <strong>{html.escape(meta['source_well'])} · {meta['source_interval_m'][0]}–{meta['source_interval_m'][1]} м</strong>
            <p>Источник: {html.escape(original)}</p>
            <label>Порода на фото <select class="label">{choices}</select></label>
            <label><input type="checkbox" class="approve"> Фото и метка совпадают, интервал верен</label>
            <label>Примечание <input class="note" placeholder="При необходимости"></label>
          </div>
        </article>''')
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Cat Island — проверка пород</title>
    <style>body{{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;background:#f5f4f0;color:#24292a}}
    h1{{margin-bottom:0}}.hint{{color:#526064}}.toolbar{{position:sticky;top:0;background:#f5f4f0;padding:12px 0;z-index:2}}
    button{{padding:10px 18px;background:#275d55;color:white;border:0;border-radius:7px;cursor:pointer}}
    .card{{display:flex;gap:20px;background:white;margin:12px 0;padding:14px;border:1px solid #d7dedb;border-radius:10px}}
    .card img{{height:300px;max-width:50%;object-fit:contain;background:#222}}
    .body{{flex:1}}label{{display:block;margin:14px 0}}select,input{{font:inherit;padding:6px}}.note{{width:75%}}</style>
    <h1>Проверка породы по фото: Cat Island</h1>
    <p class="hint">Экспериментальная выборка из {len(rows)} фрагментов. Исходный текст USGS на английском;
    русская порода получена словарём. Отмечайте только действительно совпадающие фото, интервалы и породы.
    Отложенные керны здесь не показаны.</p>
    <div class="toolbar"><button onclick="exportReview()">Скачать проверенные решения JSONL</button>
    <span id="count">0 подтверждено</span></div>{''.join(cards)}
    <script>
    const cards=[...document.querySelectorAll('.card')];
    document.addEventListener('change',()=>{{document.querySelector('#count').textContent=cards.filter(c=>c.querySelector('.approve').checked).length+' подтверждено';}});
    function exportReview(){{
      const rows=cards.filter(c=>c.querySelector('.approve').checked).map(c=>({{
        record_id:c.dataset.record,approved:true,lithology_ru:c.querySelector('.label').value,
        note:c.querySelector('.note').value.trim(),reviewed_at:new Date().toISOString()
      }}));
      const blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{{type:'application/x-ndjson'}});
      const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='cat_island_geologist_review.jsonl';link.click();
      setTimeout(()=>URL.revokeObjectURL(link.href),3000);
    }}</script></html>'''
    OUTPUT.write_text(page, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
