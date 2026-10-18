"""Record the sealed whole-well evaluation and create a geologist review queue."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "outputs/mendeley_lithology_pilot_test_evaluation.json"
ADAPTER = ROOT / "outputs/core_vlm_mendeley_lithology_pilot_lora/adapter_model.safetensors"
OUTPUT = ROOT / "outputs"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    training = json.loads((ADAPTER.parent / "training_metrics.json").read_text(encoding="utf-8"))
    inventory = json.loads((ROOT / "data/external_raw/mendeley_lithology/inventory.json").read_text(encoding="utf-8"))
    train_rows = [json.loads(line) for line in (ROOT / "data/mendeley_lithology_pilot/train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert report["split"] == "test" and report["score"]["count"] == 200
    assert not set(report["train_wells"]) & set(report["heldout_wells"])

    failures = [row for row in report["records"] if row["adapter_label"] != row["expected"]]
    queue = OUTPUT / "mendeley_lithology_pilot_error_review.csv"
    with queue.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "record_id", "well_id", "image", "source_archive_member", "source_label_ru",
            "base_prediction", "adapter_prediction", "review_status", "geologist_corrected_label", "reviewer_name", "reviewer_note",
        ])
        writer.writeheader()
        for row in failures:
            writer.writerow({
                "record_id": row["record_id"],
                "well_id": row["well_id"],
                "image": row["image"],
                "source_archive_member": row["source_archive_member"],
                "source_label_ru": row["expected"],
                "base_prediction": row["base_label"] or row["base_raw"],
                "adapter_prediction": row["adapter_label"] or row["adapter_raw"],
                "review_status": "pending",
                "geologist_corrected_label": "",
                "reviewer_name": "",
                "reviewer_note": "",
            })

    cards = []
    for row in report["records"]:
        image = Path(row["image"])
        relative = image.relative_to(ROOT).as_posix()
        correct = row["adapter_label"] == row["expected"]
        cards.append(
            '<article class="card ' + ("correct" if correct else "error") + '">'
            f'<img loading="lazy" src="../{html.escape(relative, quote=True)}" alt="Фрагмент керна">'
            f'<div><b>{html.escape(row["expected"])}</b> → {html.escape(row["adapter_label"] or row["adapter_raw"])}'
            f'<small>{html.escape(row["well_id"])} · {html.escape(row["record_id"])}</small></div>'
            '</article>'
        )
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Mendeley: проверка литологии</title>
<style>body{{font:16px system-ui;margin:2rem;background:#f5f5f3;color:#222}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}}.card{{background:white;border:2px solid #c63;border-radius:8px;overflow:hidden}}.card.correct{{border-color:#4a8}}img{{width:100%;aspect-ratio:1;object-fit:cover}}.card div{{padding:8px}}small{{display:block;color:#666;margin-top:6px}}button{{padding:8px 12px;margin-right:8px}}body.errors .correct,body.corrects .error{{display:none}}</style>
<h1>Контроль литологии Mendeley</h1><p>Отложенные скважины: {html.escape(', '.join(report['heldout_wells']))}. Источник класса — папка набора, индивидуальной геологической проверки нет. Ошибки остаются в очереди проверки; фото этих скважин не входят в следующую учебную выборку.</p>
<p>Адаптер: {report['score']['adapter_correct']}/{report['score']['count']}; база: {report['score']['base_correct']}/{report['score']['count']}. Красная рамка — ошибка.</p>
<button onclick="document.body.className='errors'">Ошибки ({len(failures)})</button><button onclick="document.body.className='corrects'">Верные</button><button onclick="document.body.className=''">Все</button>
<section class="grid">{''.join(cards)}</section></html>'''
    (OUTPUT / "mendeley_lithology_pilot_error_review.html").write_text(page, encoding="utf-8")

    card = {
        "name": "core-vlm-mendeley-lithology-pilot",
        "date": "2026-10-12",
        "status": "experimental_not_integrated",
        "task": "Russian four-class lithology/non-core classification of 2x2 cm image crops; no facies or description",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "adapter_path": str(ADAPTER.relative_to(ROOT)).replace("\\", "/"),
        "adapter_sha256": hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
        "source": {"url": inventory["source_url"], "doi": inventory["doi"], "license": inventory["license"], "archive_sha256": inventory["archive_sha256"]},
        "training": {
            "manifest": "data/mendeley_lithology_pilot/train.jsonl",
            "train_crops": training["candidate_rows_used"],
            "train_wells": training["independent_wells"],
            "train_wells_by_class": {label: len({row["metadata"]["well_id"] for row in train_rows if row["metadata"]["source_label_en"] == label}) for label in ("sandstone", "limestone", "shale", "garbage")},
            "geologist_approved_rows": training["geology_reviewed_rows_used"],
            "epochs": training["epoch"],
            "train_loss": training["train_loss"],
            "training_status": training["training_status"],
        },
        "evaluation": {
            "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "split": "sealed_whole_well_test_first_use",
            "test_wells": report["heldout_wells"],
            "test_crops": report["score"]["count"],
            "base_correct": report["score"]["base_correct"],
            "adapter_correct": report["score"]["adapter_correct"],
            "majority_baseline_correct": report["score"]["majority_correct"],
            "base_macro_f1": report["score"]["base_macro_f1"],
            "adapter_macro_f1": report["score"]["adapter_macro_f1"],
            "adapter_recall_by_class": report["score"]["adapter_recall_by_class"],
            "per_well": {well: {"correct": row["adapter_correct"], "total": row["count"]} for well, row in report["per_well"].items()},
            "error_review_csv": str(queue.relative_to(ROOT)).replace("\\", "/"),
            "error_review_html": "outputs/mendeley_lithology_pilot_error_review.html",
            "decision": "Do not integrate: 0/50 limestone and 2/50 shale despite 80/200 overall.",
        },
        "limitations": [
            "Folder labels are source-direct but individual crops were not geologist reviewed.",
            "The four-class source is limited to 2x2 cm crops and cannot train facies, depositional environment or full descriptions.",
            "The original source splits share all 28 wells; this experiment uses separate whole wells.",
            "Only five training wells contain limestone crops in the pilot; each class has fewer independent wells than image crops.",
            "Further model selection against these same four test wells would invalidate the sealed-test interpretation.",
            "A practical model needs expert-reviewed ancient-core examples and independent external wells.",
        ],
    }
    (ROOT / "model_spec_mendeley_lithology_pilot.json").write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test_correct": card["evaluation"]["adapter_correct"], "test_crops": card["evaluation"]["test_crops"], "errors_for_review": len(failures)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
