"""Record provenance and unchanged-validation metrics for 4000-crop LoRA."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data/mendeley_lithology_4000/train.jsonl"
VAL = ROOT / "data/mendeley_lithology_expanded/val.jsonl"
PREVIOUS = ROOT / "outputs/mendeley_lithology_expanded_val_evaluation.json"
CURRENT = ROOT / "outputs/mendeley_lithology_4000_val_evaluation.json"
ADAPTER = ROOT / "outputs/core_vlm_mendeley_lithology_4000_lora/adapter_model.safetensors"
SOURCE = ROOT / "data/external_raw/mendeley_lithology/inventory.json"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    train, val = rows(TRAIN), rows(VAL)
    old = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    new = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = json.loads((ADAPTER.parent / "training_metrics.json").read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert len(train) == 4000 and len(val) == 200
    assert new["split"] == "val" and new["train_crops"] == 4000
    assert new["heldout_crops"] == old["heldout_crops"] == len(val)
    assert not set(new["train_wells"]) & set(new["heldout_wells"])
    assert [(r["record_id"], r["expected"], r["base_raw"]) for r in new["records"]] == [
        (r["record_id"], r["expected"], r["base_raw"]) for r in old["records"]
    ]
    errors = [r for r in new["records"] if r["adapter_label"] != r["expected"]]
    review = ROOT / "outputs/mendeley_lithology_4000_val_errors.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["record_id", "well_id", "image", "source_archive_member", "source_label_ru",
                  "prior_prediction", "new_prediction", "review_status", "geologist_corrected_label", "reviewer_note"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        prior = {r["record_id"]: r for r in old["records"]}
        for row in errors:
            writer.writerow({
                "record_id": row["record_id"], "well_id": row["well_id"], "image": row["image"],
                "source_archive_member": row["source_archive_member"], "source_label_ru": row["expected"],
                "prior_prediction": prior[row["record_id"]]["adapter_label"] or prior[row["record_id"]]["adapter_raw"],
                "new_prediction": row["adapter_label"] or row["adapter_raw"],
                "review_status": "pending", "geologist_corrected_label": "", "reviewer_note": "",
            })
    by_class = Counter(r["metadata"]["source_label_en"] for r in train)
    confusion = {
        expected: dict(Counter(
            row["adapter_label"] or "неопределено"
            for row in new["records"] if row["expected"] == expected
        ))
        for expected in ("песчаник", "известняк", "глинистая порода", "не керн")
    }
    cards = []
    prior = {r["record_id"]: r for r in old["records"]}
    transitions = Counter()
    for row in new["records"]:
        was_correct = prior[row["record_id"]]["adapter_label"] == row["expected"]
        now_correct = row["adapter_label"] == row["expected"]
        status = "stable-correct" if was_correct and now_correct else (
            "fixed" if now_correct else "regressed" if was_correct else "stable-error"
        )
        transitions[status] += 1
        src = Path(os.path.relpath(row["image"], ROOT / "outputs")).as_posix()
        cards.append(
            f'<article class="{status}"><img loading="lazy" src="{html.escape(src)}" alt="Фрагмент керна">'
            f'<div><b>{html.escape(row["expected"])}</b><br>Прежний: '
            f'{html.escape(prior[row["record_id"]]["adapter_label"] or "неопределено")}<br>'
            f'Новый: {html.escape(row["adapter_label"] or "неопределено")}<small>'
            f'{html.escape(row["well_id"])} · {html.escape(row["record_id"])}</small></div></article>'
        )
    html_path = ROOT / "outputs/mendeley_lithology_4000_val_review.html"
    html_path.write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Mendeley: сравнение адаптеров</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#f1f3f5;color:#1f252b}'
        'section{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}'
        'article{background:white;border:3px solid #aaa;border-radius:8px;overflow:hidden}'
        '.fixed{border-color:#37a67a}.regressed{border-color:#d45d55}.stable-error{border-color:#e9aa3c}'
        'img{width:100%;aspect-ratio:1;object-fit:cover}article div{padding:8px}small{display:block;color:#667;margin-top:6px}'
        'button{padding:8px;margin:4px}body.fix .regressed,body.fix .stable-error,body.fix .stable-correct,'
        'body.reg .fixed,body.reg .stable-error,body.reg .stable-correct,'
        'body.err .fixed,body.err .regressed,body.err .stable-correct{display:none}</style>'
        '<h1>Сравнение двух адаптеров на тех же 200 кадрах</h1>'
        '<p>Зелёный — исправлено, красный — новая ошибка, оранжевый — ошибка осталась. '
        'Классы взяты из папок источника и индивидуально геологом не проверены.</p>'
        '<button onclick="document.body.className=\'fix\'">Исправлено</button>'
        '<button onclick="document.body.className=\'reg\'">Ухудшилось</button>'
        '<button onclick="document.body.className=\'err\'">Осталось ошибкой</button>'
        '<button onclick="document.body.className=\'\'">Все</button><section>'
        + "".join(cards) + '</section></html>', encoding="utf-8"
    )
    card = {
        "name": "core-vlm-mendeley-lithology-4000",
        "date": "2026-10-12",
        "status": "experimental_validation_only_not_integrated",
        "task": "Four-class lithology/non-core recognition on small core photo patches, Russian class name only",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "initial_adapter": "outputs/core_vlm_mendeley_lithology_expanded_lora/adapter_model.safetensors",
        "adapter_path": str(ADAPTER.relative_to(ROOT)).replace("\\", "/"),
        "adapter_sha256": hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
        "source": {"url": source["source_url"], "doi": source["doi"], "license": source["license"],
                   "archive_sha256": source["archive_sha256"]},
        "training": {
            "manifest": str(TRAIN.relative_to(ROOT)).replace("\\", "/"),
            "train_crops": len(train), "train_wells": len(new["train_wells"]),
            "by_source_class": dict(by_class), "geologist_approved_rows": metrics["geology_reviewed_rows_used"],
            "epochs": metrics["epoch"], "learning_rate": 3e-6,
            "train_loss": metrics["train_loss"], "runtime_sec": metrics["train_runtime"],
        },
        "validation": {
            "report": str(CURRENT.relative_to(ROOT)).replace("\\", "/"),
            "wells": new["heldout_wells"], "crops": len(val),
            "prior_correct": old["score"]["adapter_correct"],
            "new_correct": new["score"]["adapter_correct"],
            "prior_macro_f1": old["score"]["adapter_macro_f1"],
            "new_macro_f1": new["score"]["adapter_macro_f1"],
            "new_recall_by_class": new["score"]["adapter_recall_by_class"],
            "new_confusion_by_expected_class": confusion,
            "per_well": {well: {"correct": item["adapter_correct"], "total": item["count"]}
                         for well, item in new["per_well"].items()},
            "errors_csv": str(review.relative_to(ROOT)).replace("\\", "/"),
            "comparison_review_html": str(html_path.relative_to(ROOT)).replace("\\", "/"),
            "transition_counts": dict(transitions),
        },
        "limitations": [
            "The same validation set guided the previous continuation; this is development performance, not independent test accuracy.",
            "The former test wells were used once for the pilot and were not used here.",
            "Source folder labels are not individually reviewed by a geologist.",
            "Mendeley has no complete descriptions, facies, depositional environments or interval contacts.",
            "A larger number of correlated patches from the same 20 wells does not add independent wells.",
        ],
    }
    path = ROOT / "model_spec_mendeley_lithology_4000.json"
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prior": card["validation"]["prior_correct"], "new": card["validation"]["new_correct"],
                      "total": len(val), "errors": len(errors)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
