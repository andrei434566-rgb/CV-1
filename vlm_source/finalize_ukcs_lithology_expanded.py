"""Write a reproducible model card and human review artifacts after UKCS evaluation."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/ukcs_q204_q205_lithology_expanded"
ADAPTER = ROOT / "outputs/core_vlm_ukcs_lithology_expanded_lora"
REPORT = ROOT / "outputs/ukcs_lithology_expanded_evaluation.json"
CARD = ROOT / "model_spec_ukcs_lithology_expanded.json"
ERRORS = ROOT / "outputs/ukcs_lithology_expanded_errors.csv"
REVIEW = ROOT / "outputs/ukcs_lithology_expanded_review.html"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metrics = json.loads((ADAPTER / "training_metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))
    adapter_file = ADAPTER / "adapter_model.safetensors"
    rows = report["records"]
    if len(rows) != 118 or report["training_crops"] != 463 or audit["train_crops"] != 463:
        raise ValueError("Unexpected manifest or holdout size")
    if metrics["candidate_rows_used"] != 463 or metrics["training_approved_rows_used"] != 0:
        raise ValueError("Unexpected training metrics")
    with ERRORS.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["record_id", "well_id", "depth_top_m", "depth_bottom_m",
                                                  "expected_code", "expected_ru_editorial", "pilot_raw", "pilot_code",
                                                  "expanded_raw", "expanded_code", "image"])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows
                         if row["expanded_code"] != row["expected_code"])
    cards = []
    for row in rows:
        correct = row["expanded_code"] == row["expected_code"]
        src = Path(row["image"]).resolve().as_uri()
        cards.append(
            f'<article class="{"ok" if correct else "error"}">'
            f'<img src="{html.escape(src, quote=True)}" alt="Фрагмент керна">'
            f'<div><strong>{html.escape(row["record_id"])}</strong> '
            f'{html.escape(row["well_id"])}; {row["depth_top_m"]}–{row["depth_bottom_m"]} м<br>'
            f'Эталон источника: {html.escape(row["expected_code"])} — '
            f'{html.escape(row["expected_ru_editorial"])}<br>'
            f'Старый адаптер: {html.escape(row["pilot_raw"])} '
            f'[{html.escape(str(row["pilot_code"]))}]<br>'
            f'Новый адаптер: {html.escape(row["expanded_raw"])} '
            f'[{html.escape(str(row["expanded_code"]))}]</div></article>'
        )
    REVIEW.write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>UKCS: проверка расширенного адаптера</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px}'
        'article{display:flex;gap:18px;margin:14px 0;padding:12px;border:2px solid #ddd}'
        'article.error{border-color:#b64040}article.ok{border-color:#478a5b}'
        'img{max-width:300px;max-height:450px;object-fit:contain}div{line-height:1.7}</style>'
        '<h1>UKCS: 118 контрольных фрагментов</h1>'
        '<p>Исходные метки и русские названия ещё не проверены геологом. '
        'Красная рамка обозначает ошибку нового адаптера относительно исходного кода.</p>'
        + ''.join(cards) + '</html>', encoding="utf-8"
    )
    card = {
        "name": "core-vlm-ukcs-lithology-expanded",
        "date": "2026-10-12",
        "status": "experimental_validation_only_not_integrated",
        "task": "Five source-aligned UKCS lithology classes on core image crops; Russian class name only",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "initial_adapter": "outputs/core_vlm_ukcs_lithology_pilot_lora/adapter_model.safetensors",
        "adapter_path": str(adapter_file.relative_to(ROOT)).replace("\\", "/"),
        "adapter_sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
        "source": {
            "data_url": "https://figshare.com/articles/dataset/UKCS_Q204_Q205_Subsurface_Data_products_for_Machine_Learning_Study/14265689",
            "paper_url": "https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2021.659611/full",
            "bgs_terms": "https://www.bgs.ac.uk/information-hub/photos-and-images/",
            "archive_sha256": "2957abe45d046c7206b4c8c55d769fceb6aa26eedf20f8f29ceb8f28211ecf41",
            "class_mapping": report["source_class_mapping"],
        },
        "training": {
            "manifest": "data/ukcs_q204_q205_lithology_expanded/train_source_terminology.jsonl",
            "crops": 463, "wells": 9, "geologist_approved_rows": 0,
            "epochs": metrics.get("epoch"), "train_loss": metrics.get("train_loss"),
            "runtime_sec": metrics.get("train_runtime"),
            "source_label_rows_verified": audit["source_label_rows_verified"],
        },
        "validation": {
            "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "fixed_holdout_crops": 118, "wells": report["holdout_wells"],
            "tile_level": report["tile_level"],
            "source_run_level": report["source_run_level"],
            "per_well": report["per_well"],
            "errors_csv": str(ERRORS.relative_to(ROOT)).replace("\\", "/"),
            "review_html": str(REVIEW.relative_to(ROOT)).replace("\\", "/"),
        },
        "limitations": report["limitations"],
    }
    CARD.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pilot_correct": report["tile_level"]["pilot_correct"],
                      "expanded_correct": report["tile_level"]["expanded_correct"],
                      "pilot_f1": report["tile_level"]["pilot_macro_f1"],
                      "expanded_f1": report["tile_level"]["expanded_macro_f1"],
                      "adapter_sha256": card["adapter_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
