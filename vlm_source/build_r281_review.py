"""Build an offline Russian review page for R-281 photo/description candidates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/r281_slices/records.jsonl"
OUTPUT = ROOT / "outputs/r281_geologist_review.html"
DATA_KEY = "r281-geologist-review-v1"
ROWS = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
DISPLAY_ROWS = [{
    "id": row["record_id"],
    "core": row["core_number"],
    "depth": row["depth_m"],
    "image": "../data/r281_slices/images/" + Path(row["image"]).name,
    "description": row["description"],
    "suggested_rock": row["rock_first_word"],
} for row in ROWS]
EMBEDDED = json.dumps(DISPLAY_ROWS, ensure_ascii=False).replace("<", "\\u003c")


def main() -> None:
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка фото и описаний 281Р</title>
<style>
:root{{--bg:#101a1c;--panel:#17282b;--text:#edf4ee;--muted:#b9c8c3;--accent:#aee0b3;--line:#416061}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,Arial,sans-serif}}
header{{position:sticky;top:0;background:#101a1cf2;padding:16px 24px;border-bottom:1px solid var(--line);z-index:2}}
h1{{font-size:1.45rem;margin:0 0 4px}}p{{margin:6px 0;color:var(--muted)}}.toolbar{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:12px}}
button,select,input,textarea{{font:inherit;border:1px solid var(--line);border-radius:7px;background:#21383b;color:var(--text);padding:7px 9px}}
button{{cursor:pointer;background:#34594d}}button:hover{{background:#407057}}button:focus,select:focus,input:focus,textarea:focus{{outline:2px solid var(--accent)}}
main{{max-width:1360px;margin:auto;padding:18px 24px}}.card{{display:grid;grid-template-columns:minmax(220px,360px) 1fr;gap:18px;padding:16px;margin-bottom:18px;background:var(--panel);border:1px solid var(--line);border-radius:12px}}
.card img{{width:100%;max-height:300px;object-fit:contain;background:#091113;border-radius:7px}}.meta{{font-weight:700;color:var(--accent)}}.source{{white-space:pre-wrap;margin:8px 0 14px}}
.fields{{display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:11px}}label{{display:block;color:var(--muted);font-size:.9rem}}label>*{{display:block;width:100%;margin-top:4px}}
textarea{{resize:vertical;min-height:72px}}.wide{{grid-column:1/-1}}.check{{display:flex;gap:7px;align-items:center;color:var(--text)}}.check input{{width:auto;margin:0}}.note{{font-size:.9rem}}
@media(max-width:760px){{.card{{grid-template-columns:1fr}}.fields{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>281Р: проверка 152 пар «фото — описание»</h1>
<p>Фото торца низкого разрешения. Подтвердите только то, что действительно видно; обстановку указывайте при наличии геологических оснований. Данные сохраняются в этом браузере и экспортируются в JSONL.</p>
<div class="toolbar"><span id="progress"></span><label>Керн <select id="core-filter"><option value="">Все</option></select></label>
<label><input type="checkbox" id="only-unreviewed"> Только непроверенные</label>
<button id="export">Скачать разметку JSONL</button><label>Загрузить прежнюю разметку <input id="import" type="file" accept=".jsonl,.txt,application/json"></label></div></header>
<main id="cards"></main>
<script>
const rows={EMBEDDED};const key={json.dumps(DATA_KEY)};let reviews=JSON.parse(localStorage.getItem(key)||'{{}}');
const rocks=['неопределено','песчаник','алевролит','аргиллит','уголь','переслаивание','другое'];
const settings=['неопределено','русло','прирусловая зона','пойма','болото','дельта','озеро','морская или прибрежная','другая'];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const opts=(items,current)=>'<option value="">Не проверено</option>'+items.map(x=>`<option value="${{esc(x)}}" ${{x===current?'selected':''}}>${{esc(x)}}</option>`).join('');
function persist(){{localStorage.setItem(key,JSON.stringify(reviews));progress()}}
function progress(){{const n=rows.filter(r=>reviews[r.id]?.reviewed).length;document.getElementById('progress').textContent=`Проверено: ${{n}} / ${{rows.length}}`;}}
function render(){{const selected=document.getElementById('core-filter').value;const pending=document.getElementById('only-unreviewed').checked;
document.getElementById('cards').innerHTML=rows.filter(r=>(!selected||String(r.core)===selected)&&(!pending||!reviews[r.id]?.reviewed)).map(r=>{{const v=reviews[r.id]||{{}};return `<article class="card" data-id="${{esc(r.id)}}"><div><img loading="lazy" src="${{esc(r.image)}}" alt="Фото ${{esc(r.id)}}"><p class="note">${{esc(r.id)}} · торец керна, глубина ${{esc(r.depth)}} м</p></div><div><div class="meta">Керн №${{esc(r.core)}} · ${{esc(r.depth)}} м</div><div class="source"><strong>Текст отчёта:</strong> ${{esc(r.description)}}</div><p class="note">Из текста автоматически извлечено: ${{esc(r.suggested_rock)}}. Это не проверенная визуальная метка.</p><div class="fields"><label>Порода на фото<select data-field="rock_type">${{opts(rocks,v.rock_type)}}</select></label><label>Обстановка осадконакопления<select data-field="environment">${{opts(settings,v.environment)}}</select></label><label class="wide">Исправленное описание видимых признаков<textarea data-field="visual_description">${{esc(v.visual_description||'')}}</textarea></label><label class="wide">Геологическое основание обстановки и замечания<textarea data-field="evidence">${{esc(v.evidence||'')}}</textarea></label><label class="check"><input type="checkbox" data-field="photo_usable" ${{v.photo_usable?'checked':''}}> Фото пригодно для обучения</label><label class="check"><input type="checkbox" data-field="reviewed" ${{v.reviewed?'checked':''}}> Проверка завершена</label></div></div></article>`}}).join('');progress()}}
const coreFilter=document.getElementById('core-filter');[...new Set(rows.map(r=>r.core))].sort((a,b)=>a-b).forEach(n=>coreFilter.add(new Option('№'+n,n)));
coreFilter.onchange=render;document.getElementById('only-unreviewed').onchange=render;
document.getElementById('cards').addEventListener('input',e=>{{const field=e.target.dataset.field;if(!field)return;const id=e.target.closest('.card').dataset.id;reviews[id]||={{}};reviews[id][field]=e.target.type==='checkbox'?e.target.checked:e.target.value;persist()}});
document.getElementById('cards').addEventListener('change',e=>{{if(e.target.dataset.field==='reviewed'&&document.getElementById('only-unreviewed').checked)render()}});
document.getElementById('export').onclick=()=>{{const text=rows.map(r=>JSON.stringify({{record_id:r.id,well_id:'281Р',core_number:r.core,depth_m:r.depth,review:reviews[r.id]||{{}}}})).join('\\n')+'\\n';const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{{type:'application/jsonl;charset=utf-8'}}));a.download='review_281R.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000)}};
document.getElementById('import').onchange=async e=>{{const f=e.target.files[0];if(!f)return;const text=await f.text();for(const line of text.split(/\\r?\\n/)){{if(!line.trim())continue;const row=JSON.parse(line);if(rows.some(r=>r.id===row.record_id)&&row.review)reviews[row.record_id]=row.review}}persist();render()}};
render();
</script></body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "records": len(ROWS), "bytes": OUTPUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
