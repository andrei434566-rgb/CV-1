"""Build a human review queue from held-out binary lithology evaluations.

The source labels and crop alignment are provisional. The generated review form
never promotes edited rows into the training set automatically.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MIX = json.loads((OUT / "multidomain_binary_pilot_evaluation.json").read_text(encoding="utf-8"))
UKCS = json.loads((OUT / "ukcs_lithology_binary_transfer_evaluation.json").read_text(encoding="utf-8"))
BAR = json.loads((OUT / "barnegat_binary_pilot_evaluation.json").read_text(encoding="utf-8"))


def read_jsonl(relative: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / relative).read_text(encoding="utf-8").splitlines() if line]


def vote(labels: list[str | None]) -> str:
    counts = Counter(label for label in labels if label)
    if not counts:
        return "не определено"
    highest = max(counts.values())
    winners = [label for label, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 else "спорное голосование"


def src(path: str) -> str:
    image = Path(path)
    if not image.is_absolute():
        image = ROOT / image
    try:
        return "../" + image.relative_to(ROOT).as_posix()
    except ValueError:
        return image.as_uri()


def priority(row: dict) -> int:
    if row["mixed_prediction"] != row["expected"]:
        return 1
    if row["other_prediction"] != row["expected"]:
        return 2
    return 3


def make_rows() -> list[dict]:
    rows = []
    mixed_external = MIX["external_records"]
    ukcs_external = {item["record_id"]: item for item in UKCS["external_records"]}
    r41_meta = {str(item["metadata"]["source_row"]): item["metadata"] for item in read_jsonl("data/v8_single_column/test_shared_v7_rows.jsonl")}
    cat_meta = {item["metadata"]["record_id"]: item["metadata"] for item in read_jsonl("data/cat_island_four_class_pilot/heldout.jsonl")}
    bar_meta = {item["metadata"]["record_id"]: item["metadata"] for item in read_jsonl("data/barnegat_binary_pilot/holdout_candidates.jsonl")}
    bar_old = {item["record_id"]: item for item in BAR["results"]}

    r41 = defaultdict(list)
    for item in mixed_external:
        if item["dataset"] == "r41_ancient_core":
            r41[item["group"]].append(item)
    for group, items in r41.items():
        meta = r41_meta[group]
        rows.append({
            "id": f"r41_row_{group}", "dataset": "Р‑41: древний керн",
            "source_unit": f"Строка {group}", "well": str(meta.get("source_well") or "Р‑41"),
            "depth": str(meta.get("source_description_interval_m") or ""),
            "expected": items[0]["expected"],
            "source_lithology": items[0]["source_label"],
            "source_description": meta.get("source_description_full", ""),
            "mixed_prediction": vote([i["adapter_label"] for i in items]),
            "other_prediction": vote([ukcs_external[i["record_id"]]["adapter_label"] for i in items]),
            "other_model": "UKCS 5 классов → 2",
            "images": [i["image"] for i in items],
            "source_status": "Пара фото и описания подтверждена пользователем; привязка каждого фрагмента к глубине требует проверки.",
        })

    for item in mixed_external:
        if item["dataset"] != "cat_island_modern_core":
            continue
        meta = cat_meta[item["record_id"]]
        rows.append({
            "id": item["record_id"], "dataset": "Cat Island: современный керн",
            "source_unit": item["record_id"], "well": str(meta.get("source_well") or item["group"]),
            "depth": str(meta.get("source_interval_m") or ""),
            "expected": item["expected"],
            "source_lithology": meta.get("lithology_source_en", item.get("source_label", "")),
            "source_description": meta.get("source_description_en", ""),
            "mixed_prediction": item["adapter_label"],
            "other_prediction": ukcs_external[item["record_id"]]["adapter_label"],
            "other_model": "UKCS 5 классов → 2",
            "images": [item["image"]],
            "source_status": "Литология перенесена из опубликованного описания; фрагмент и границы требуют проверки.",
        })

    bar_groups = defaultdict(list)
    for item in MIX["heldout_records"]:
        if item["dataset"] == "barnegat":
            bar_groups[item["group"]].append(item)
    for group, items in bar_groups.items():
        first_meta = bar_meta[items[0]["record_id"]]
        rows.append({
            "id": group, "dataset": "Barnegat Bay: современный керн",
            "source_unit": group, "well": items[0]["well_id"],
            "depth": str(first_meta.get("depth_interval_cm") or "") + " см",
            "expected": items[0]["expected"],
            "source_lithology": first_meta.get("source_lithology", ""),
            "source_description": first_meta.get("source_description", ""),
            "mixed_prediction": vote([i["adapter_label"] for i in items]),
            "other_prediction": vote([bar_old[i["record_id"]]["adapter_label"] for i in items]),
            "other_model": "Barnegat pilot",
            "images": [i["image"] for i in items],
            "source_status": "Источник USGS; границы фрагмента и двоичное отображение литологии требуют проверки.",
        })

    for row in rows:
        row["priority"] = priority(row)
        row["reason"] = (
            "Ошибка совместной модели" if row["priority"] == 1
            else "Расхождение с альтернативной моделью" if row["priority"] == 2
            else "Контрольный пример"
        )
    return sorted(rows, key=lambda row: (row["priority"], row["dataset"], row["id"]))


def write_csv(rows: list[dict]) -> Path:
    path = OUT / "geologist_priority_review_20261011.csv"
    fields = ["priority", "reason", "dataset", "id", "well", "source_unit", "depth", "expected", "source_lithology", "source_description", "mixed_prediction", "other_model", "other_prediction", "source_status", "images", "photo_depth_alignment_ok", "corrected_lithology", "visible_features", "depositional_environment", "corrected_russian_description", "reviewer_notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "images": " | ".join(row["images"]), "photo_depth_alignment_ok": "", "corrected_lithology": "", "visible_features": "", "depositional_environment": "", "corrected_russian_description": "", "reviewer_notes": ""})
    return path


def write_html(rows: list[dict]) -> Path:
    path = OUT / "geologist_priority_review_20261011.html"
    cards = []
    for row in rows:
        key = html.escape(row["id"], quote=True)
        photos = "".join(f'<img loading="lazy" src="{html.escape(src(image), quote=True)}" alt="Фрагмент {key}">' for image in row["images"])
        cards.append(f'''<article class="card" data-id="{key}" data-priority="{row["priority"]}" data-dataset="{html.escape(row["dataset"], quote=True)}">
<header><strong>Приоритет {row["priority"]}: {html.escape(row["reason"])}</strong><span>{html.escape(row["dataset"])} · {html.escape(row["source_unit"])}</span></header>
<div class="facts">Скважина: {html.escape(row["well"])} · Глубина: {html.escape(row["depth"])}<br>Источник: <b>{html.escape(row["expected"])}</b> ({html.escape(row["source_lithology"])})<br>Совместная модель: <b>{html.escape(row["mixed_prediction"])}</b> · {html.escape(row["other_model"])}: {html.escape(row["other_prediction"])}</div>
<div class="photos">{photos}</div>
<details><summary>Исходное описание и статус</summary><p>{html.escape(row["source_description"])}</p><small>{html.escape(row["source_status"])}</small></details>
<div class="fields"><label>Привязка фото к глубине<select data-field="photo_depth_alignment_ok"><option value="">Не проверено</option><option value="да">Да</option><option value="нет">Нет</option><option value="неясно">Неясно</option></select></label>
<label>Правильная литология<input data-field="corrected_lithology" placeholder="Например: песчаник"></label>
<label>Видимые признаки<textarea data-field="visible_features" rows="2" placeholder="Зернистость, цвет, текстура, слоистость, контакты"></textarea></label>
<label>Обстановка осадконакопления<textarea data-field="depositional_environment" rows="2" placeholder="Укажите только при наличии оснований"></textarea></label>
<label>Исправленное описание по-русски<textarea data-field="corrected_russian_description" rows="3"></textarea></label>
<label>Примечания<textarea data-field="reviewer_notes" rows="2"></textarea></label></div></article>''')
    counts = Counter(row["priority"] for row in rows)
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Очередь проверки керна</title>
<style>body{font:16px/1.45 system-ui,sans-serif;margin:0;background:#f3f5f4;color:#14251f}main{max-width:1250px;margin:auto;padding:24px}h1{margin:0 0 12px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;background:white;padding:14px;border-radius:10px;position:sticky;top:0;z-index:2;box-shadow:0 1px 5px #ccc}button,select,input,textarea{font:inherit;padding:8px;border:1px solid #9aaea4;border-radius:6px}button{background:#1c6148;color:#fff;cursor:pointer}.card{background:white;border-left:6px solid #ad5b17;padding:18px;margin:18px 0;border-radius:8px;box-shadow:0 1px 5px #d8dcd8}.card[data-priority="2"]{border-color:#bc9a2a}.card[data-priority="3"]{border-color:#6aa183}header{display:flex;gap:14px;flex-wrap:wrap;justify-content:space-between}.facts{margin:12px 0}.photos{display:flex;gap:8px;overflow:auto;max-height:380px}.photos img{max-height:360px;max-width:42%;object-fit:contain;background:#eee}details{margin:12px 0}details p{white-space:pre-wrap}.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}.fields label{display:flex;flex-direction:column;gap:4px}.fields textarea{resize:vertical}.hidden{display:none}</style>
<main><h1>Проверка ошибочных и спорных интервалов</h1><p>Черновые метки из источников и автоматические вырезки требуют геологической проверки. Исправления хранятся только в этом браузере; кнопка «Скачать исправления» сохраняет JSON. Обучение не запускается автоматически.</p>
<div class="toolbar"><label>Набор <select id="dataset"><option value="">Все</option><option>Р‑41: древний керн</option><option>Cat Island: современный керн</option><option>Barnegat Bay: современный керн</option></select></label><label>Приоритет <select id="priority"><option value="">Все</option><option value="1">1 — ошибка</option><option value="2">2 — расхождение</option><option value="3">3 — контроль</option></select></label><button id="download">Скачать исправления JSON</button><span id="count"></span></div>
<p>Всего: COUNT_TOTAL; ошибки совместной модели: COUNT_1; расхождения: COUNT_2; контрольные: COUNT_3.</p>
CARDS
<script>const cards=[...document.querySelectorAll('.card')], datasetFilter=document.getElementById('dataset'), priorityFilter=document.getElementById('priority'), counter=document.getElementById('count'), downloadButton=document.getElementById('download');const storeKey='core-geologist-review-20261011';let saved={};try{saved=JSON.parse(localStorage.getItem(storeKey)||'{}')}catch(e){}for(const card of cards){const id=card.dataset.id;for(const field of card.querySelectorAll('[data-field]')){field.value=saved[id]?.[field.dataset.field]||'';field.addEventListener('input',()=>{saved[id]??={};saved[id][field.dataset.field]=field.value;try{localStorage.setItem(storeKey,JSON.stringify(saved))}catch(e){}})}}function filter(){let n=0;for(const card of cards){const ok=(!datasetFilter.value||card.dataset.dataset===datasetFilter.value)&&(!priorityFilter.value||card.dataset.priority===priorityFilter.value);card.classList.toggle('hidden',!ok);n+=ok}counter.textContent=`Показано: ${n}`}datasetFilter.addEventListener('change',filter);priorityFilter.addEventListener('change',filter);filter();downloadButton.addEventListener('click',()=>{const blob=new Blob([JSON.stringify({schema:'core_geologist_review_v1',created_at:new Date().toISOString(),corrections:saved},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='geologist_corrections.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)})</script></main></html>'''
    document = document.replace("COUNT_TOTAL", str(len(rows))).replace("COUNT_1", str(counts[1])).replace("COUNT_2", str(counts[2])).replace("COUNT_3", str(counts[3])).replace("CARDS", "\n".join(cards))
    path.write_text(document, encoding="utf-8")
    return path


if __name__ == "__main__":
    review_rows = make_rows()
    assert len(review_rows) == 63, len(review_rows)
    assert len({row["id"] for row in review_rows}) == len(review_rows)
    csv_path = write_csv(review_rows)
    html_path = write_html(review_rows)
    print(json.dumps({"rows": len(review_rows), "priority_counts": dict(Counter(r["priority"] for r in review_rows)), "csv": str(csv_path), "html": str(html_path)}, ensure_ascii=True))
