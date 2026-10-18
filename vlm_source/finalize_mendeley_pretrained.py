"""Publish provenance, metrics, and error review for the best pretrained CV pilot."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "outputs/core_cv_mendeley_resnet18_imagenet_8000"
PRETRAINED = ROOT / "model_cache/torchvision/resnet18-f37072fd.pth"
SCRATCH = ROOT / "outputs/core_cv_mendeley_resnet18_8000/pretrained_holdout_test_evaluation.json"
VLM = ROOT / "outputs/mendeley_vlm_8000_pretrained_holdout_test.json"
CARD = ROOT / "model_spec_mendeley_resnet18_imagenet_8000.json"
ERRORS = MODEL / "test_errors.csv"
REVIEW = MODEL / "test_error_review.html"
RU = {"sandstone": "песчаник", "limestone": "известняк",
      "shale": "глинистая порода", "garbage": "не керн"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def main() -> None:
    history, val, test, scratch, vlm = (
        read(MODEL / "training_history.json"), read(MODEL / "val_evaluation.json"),
        read(MODEL / "test_evaluation.json"), read(SCRATCH), read(VLM)
    )
    if (history["best_epoch"] != 5 or val["score"]["correct"] != 370
            or test["score"]["correct"] != 369 or len(test["records"]) != 400):
        raise ValueError("Unexpected experiment or scores")
    if test["heldout_wells"] != scratch["heldout_wells"] or test["heldout_wells"] != vlm["heldout_wells"]:
        raise ValueError("Different test wells")
    scratch_by_id = {row["record_id"]: row for row in scratch["records"]}
    vlm_by_id = {row["record_id"]: row for row in vlm["records"]}
    for row in test["records"]:
        record_id = row["record_id"]
        if (row["source_archive_member"] != scratch_by_id[record_id]["source_archive_member"]
                or row["source_archive_member"] != vlm_by_id[record_id]["source_archive_member"]
                or RU[row["expected_en"]] != vlm_by_id[record_id]["expected"]):
            raise ValueError(f"Source test row mismatch: {record_id}")
    errors = [row for row in test["records"] if row["expected_en"] != row["predicted_en"]]
    with ERRORS.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["record_id", "well_id", "expected_en", "predicted_en",
                                                  "model_softmax_max_uncalibrated", "source_archive_member", "image"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in errors)
    blocks = []
    for row in errors:
        uri = Path(row["image"]).resolve().as_uri()
        blocks.append(
            '<article><img src="' + html.escape(uri, quote=True) + '" alt="фото керна">'
            '<div><strong>' + html.escape(row["record_id"]) + '</strong><br>'
            'Скважина: ' + html.escape(row["well_id"]) + '<br>'
            'Источник: ' + html.escape(row["expected_en"]) + ' — '
            + html.escape(RU[row["expected_en"]]) + '<br>'
            'Прогноз: ' + html.escape(row["predicted_en"]) + ' — '
            + html.escape(RU[row["predicted_en"]]) + '<br>'
            'Файл в архиве: ' + html.escape(row["source_archive_member"]) + '</div></article>'
        )
    REVIEW.write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Ошибки ResNet-18</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 18px}'
        'article{display:flex;gap:18px;border:1px solid #b55;padding:12px;margin:12px 0}'
        'img{width:220px;height:220px;object-fit:contain}div{line-height:1.7}</style>'
        '<h1>Ошибки на 400 новых кадрах Mendeley</h1>'
        '<p>Метки источника не проверены геологом по каждому фото. '
        'Показаны только ошибки выбранной модели; уверенность softmax не откалибрована.</p>'
        + ''.join(blocks) + '</html>', encoding="utf-8"
    )
    card = {
        "name": "core-cv-mendeley-resnet18-imagenet-8000",
        "date": "2026-10-12",
        "status": "experimental_small_crop_classifier_not_integrated",
        "task": "Four Mendeley lithology/non-core folder classes on 2x2cm core photo crops",
        "architecture": history["architecture"],
        "pretrained_weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "pretrained_weights_sha256": sha(PRETRAINED),
        "checkpoint": str((MODEL / "model.pt").relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha(MODEL / "model.pt"),
        "source": {"url": "https://data.mendeley.com/datasets/b6rj29mrss/1",
                   "doi": "10.17632/b6rj29mrss.1", "license": "CC BY 4.0",
                   "class_mapping_ru": RU},
        "training": {"images": 8000, "wells": 20, "selected_epoch": history["best_epoch"],
                     "selection_metric": "validation_macro_f1",
                     "geologist_approved_images": 0,
                     "hyperparameters": history["hyperparameters"],
                     "data_audit": history["data_audit"]},
        "validation": {"images": 400, "wells": val["heldout_wells"],
                       "score": val["score"],
                       "report": str((MODEL / "val_evaluation.json").relative_to(ROOT)).replace("\\", "/")},
        "test": {"images": 400, "wells": test["heldout_wells"],
                 "score": test["score"],
                 "accuracy_95pct_wilson_interval": wilson(test["score"]["correct"], 400),
                 "scratch_same_images_correct": scratch["score"]["correct"],
                 "qwen_same_images_correct": vlm["score"]["adapter_correct"],
                 "report": str((MODEL / "test_evaluation.json").relative_to(ROOT)).replace("\\", "/"),
                 "error_count": len(errors),
                 "errors_csv": str(ERRORS.relative_to(ROOT)).replace("\\", "/"),
                 "error_review_html": str(REVIEW.relative_to(ROOT)).replace("\\", "/")},
        "limitations": [
            "The 400 test images were unused, but the same four well identities appeared in previous research runs; no independent new-well claim.",
            "Source folder labels are not individually checked by a geologist; any mislabeled crop affects the reported score.",
            "The 95% Wilson interval treats image crops as independent and is optimistic if neighboring crops are correlated.",
            "The model expects a small crop, not a full core box or raw well photo.",
            "This is lithology/non-core classification, not full Russian description, facies or depositional environment.",
            "The model is not integrated into the geologist application and has not been tested on ancient core or local E: material.",
        ],
    }
    CARD.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"val_correct": val["score"]["correct"],
                      "test_correct": test["score"]["correct"],
                      "scratch_same_test_correct": scratch["score"]["correct"],
                      "vlm_same_test_correct": vlm["score"]["adapter_correct"],
                      "errors": len(errors), "model_sha256": card["checkpoint_sha256"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
