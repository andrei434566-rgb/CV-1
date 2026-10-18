"""Render the fixed DS1061 holdout predictions as a local review page."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "usgs_ds1061_binary_research_evaluation.json"
OUTPUT = ROOT / "outputs" / "usgs_ds1061_binary_research_review.html"


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    summary = data["summary"]
    cards = []
    for row in data["results"]:
        image_rel = "../" + str(Path(row["image"]).relative_to(ROOT)).replace("\\", "/")
        base_ok = row["base_label"] == row["expected"]
        tuned_ok = row["adapter_label"] == row["expected"]
        cards.append(
            f"<article class='{'bad' if not tuned_ok else 'good'}'>"
            f"<h3>Участок {html.escape(row['site_group'])} · {html.escape(row['record_id'])}</h3>"
            f"<img src='{html.escape(image_rel)}' alt='Фото керна'>"
            f"<p>Подпись источника: <b>{html.escape(row['source_label_en'])}</b> → {html.escape(row['expected'])}</p>"
            f"<p>Базовая Qwen: <strong>{html.escape(row['base_label'] or 'неопределено')}</strong> {'✓' if base_ok else '✗'}<br>"
            f"Новый LoRA: <strong>{html.escape(row['adapter_label'] or 'неопределено')}</strong> {'✓' if tuned_ok else '✗'}</p>"
            "</article>"
        )
    page = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS DS1061 · сравнение Qwen</title>
<style>body{font:16px system-ui;color:#24292e;background:#f4f2ee;margin:2rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:1rem}article{background:#fff;padding:1rem;border-radius:8px;border-top:5px solid #718096}article.good{border-color:#498468}article.bad{border-color:#b96555}img{width:100%;height:180px;object-fit:contain;background:#222}h3{font-size:1rem;overflow-wrap:anywhere}p{line-height:1.4}.note{background:#fff4dc;padding:1rem}</style>
<h1>USGS DS1061: базовая Qwen и новый LoRA</h1>"""
    page += (f"<p>Контроль: {summary['heldout_unique_images']} фото из целиком отложенных участков 2, 5, 12. "
             f"Базовая модель: {summary['base']['correct']}/30; новый адаптер: {summary['adapter']['correct']}/30; "
             f"постоянный ответ «глинистый»: {summary['majority_correct_on_holdout']}/30.</p>")
    page += "<p class='note'>Это исследовательские приблизительно привязанные метки из PDF. Совпадение с подписью источника не означает геологического утверждения по фото.</p><main>"
    page += "\n".join(cards) + "</main></html>"
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"{OUTPUT}: {len(cards)} records, adapter wrong {sum(r['adapter_label'] != r['expected'] for r in data['results'])}")


if __name__ == "__main__":
    main()
