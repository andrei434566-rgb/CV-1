"""Build a self-contained comparison sheet for a geologist's visual review."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "v8_evaluation"
MODELS = ["base", "v5_seen_rows", "v7_old_crops", "v8_corrected"]
LABELS = {
    "base": "Базовая Qwen",
    "v5_seen_rows": "v5 (эти строки встречались при обучении)",
    "v7_old_crops": "v7 (старая нарезка)",
    "v8_corrected": "v8 (одна колонка)",
}


def records(name: str) -> dict[int, dict]:
    return {
        row["source_row"]: row
        for line in (OUT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))
    }


def thumbnail(path: str) -> str:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail((420, 650))
        stream = io.BytesIO()
        image.save(stream, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def main() -> None:
    data = {name: records(name) for name in MODELS}
    ids = sorted(data["v8_corrected"])
    if any(set(rows) != set(ids) for rows in data.values()):
        raise ValueError("Evaluation files have different row sets")
    esc = lambda value: html.escape(str(value))
    parts = ["""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Проверка VLM на 19 интервалах Р-41</title><style>
body{font:15px/1.45 Arial,sans-serif;color:#1c2c34;margin:2rem auto;max-width:1280px;padding:0 1rem;background:#f4f7f8}
h1{font-size:1.7rem}h2{margin:0 0 .8rem}.note{padding:1rem;background:#fff0d5;border-left:4px solid #ab7317}
article{background:white;border:1px solid #cad8de;border-radius:10px;margin:1.3rem 0;padding:1rem;break-inside:avoid}
.grid{display:grid;grid-template-columns:minmax(220px,360px) 1fr;gap:1rem}img{width:100%;max-height:620px;object-fit:contain;background:#ddd}
.answers{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7rem}.answer{background:#f3f7f8;padding:.7rem;border-radius:7px}
.answer b{display:block;margin-bottom:.4rem}textarea{width:100%;min-height:5rem;box-sizing:border-box;font:inherit}input{font:inherit}
button{background:#135d61;color:white;border:0;border-radius:6px;padding:.7rem 1rem;cursor:pointer}
@media(max-width:760px){.grid,.answers{display:block}.answer{margin:.5rem 0}}@media print{button{display:none}body{background:white}article{border-color:#999}}
</style></head><body><h1>Сравнение описаний керна Р-41</h1>
<p class="note">19 отложенных строк одной скважины; для каждой показан один фотофрагмент. Эти оценки не подтверждают правильность геологического описания. v5 обучалась на этих строках и показана только для сравнения. Глубина, минеральный состав и текстура требуют сверки с исходным описанием и изображением. Галочка проверки не включает ответ автоматически в обучение.</p>
<p>Базовая модель и v7/v8 не обучались на этих 19 строках. Совпадение ключевых слов породы (F1): база 0,15; v7 0,34; v8 0,45. По цвету: 0,62; 0,68; 0,52. Выдуманные детали встречаются у всех моделей.</p>
<p><label>Геолог: <input id="reviewer" placeholder="ФИО"></label> <button onclick="downloadReviews()">Скачать введённые исправления JSON</button></p>"""]
    for source_row in ids:
        source = data["v8_corrected"][source_row]
        top, base = source["interval_m"]
        parts.append(f'<article data-row="{source_row}"><h2>Строка {source_row}, {top:.3f}–{base:.3f} м</h2><div class="grid">')
        parts.append(f'<div><img src="{thumbnail(source["image"])}" alt="Керн, строка {source_row}"></div><div>')
        parts.append(f'<p><b>Исходное описание слоя:</b> {esc(source["source_description_full"])}</p><div class="answers">')
        for name in MODELS:
            row = data[name][source_row]
            status = "" if row["valid_json"] else " — ответ невалидный JSON"
            parts.append(f'<div class="answer"><b>{esc(LABELS[name] + status)}</b>{esc(row["predicted_description"])}</div>')
        parts.append('</div><p><label>Исправленное описание видимого участка:<br><textarea class="correction"></textarea></label></p>')
        parts.append('<p><label>Примечание к фото и глубинной привязке:<br><textarea class="notes"></textarea></label></p>')
        parts.append('<label><input type="checkbox" class="reviewed"> Проверено геологом</label></div></div></article>')
    parts.append("""<script>
function downloadReviews(){
 const reviewer=document.getElementById('reviewer').value.trim();
 const rows=[...document.querySelectorAll('article')].map(a=>({
  source_row:Number(a.dataset.row), reviewer,
  corrected_description:a.querySelector('.correction').value.trim(),
  notes:a.querySelector('.notes').value.trim(),
  geologist_reviewed:a.querySelector('.reviewed').checked,
  training_approved:false
 }));
 const blob=new Blob([JSON.stringify({source:'v8_heldout_review',rows},null,2)],{type:'application/json'});
 const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='r41_v8_geologist_review.json';link.click();
 setTimeout(()=>URL.revokeObjectURL(link.href),1000);
}
</script></body></html>""")
    destination = OUT / "review_19.html"
    destination.write_text("\n".join(parts), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
