"""Train a separate local CV candidate from an audited external dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _deny_network(*args, **kwargs):
    raise ConnectionError("Сетевые загрузки отключены в локальном процессе обучения CV")


def install_offline_network_guard() -> None:
    """Apply only inside the dedicated trainer process, before heavy imports."""
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline")
    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network


def configure_offline_ultralytics() -> None:
    """Avoid implicit font/model downloads, including checks with plots=False.

    Ultralytics calls check_font during dataset validation independently of
    plots. A bundled Matplotlib font is sufficient for optional local labels;
    with no local font PIL's existing default fallback remains available.
    """
    try:
        matplotlib = importlib.import_module("matplotlib")
        bundled_font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    except (ImportError, AttributeError):
        bundled_font = None

    def local_font(font="Arial.ttf"):
        requested = Path(font)
        if requested.is_file():
            return requested
        if bundled_font is not None and bundled_font.is_file():
            return bundled_font
        return None

    for name in ("ultralytics.utils.checks", "ultralytics.data.utils", "ultralytics.utils.plotting"):
        module = importlib.import_module(name)
        module.check_font = local_font
    downloads = importlib.import_module("ultralytics.utils.downloads")
    downloads.safe_download = _deny_network
    downloads.download = _deny_network
    # These helpers are imported by value in the dataset module.
    data = importlib.import_module("ultralytics.data.utils")
    data.safe_download = _deny_network
    data.download = _deny_network


def prepared_training_input(folder: Path, classification_root: Path | None = None) -> dict:
    folder = folder.resolve()
    summary_path = folder / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Не найден паспорт подготовленного набора. Сначала подготовьте материалы в приложении.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task = summary.get("task")
    if task not in {"segmentation", "classification"}:
        raise ValueError("Этому набору нужна проверка пар фото–описание; обучение CV по нему пока недоступно.")
    if summary.get("readiness") != "ready":
        raise ValueError("В наборе не готов независимый контроль. Откройте отчёт подготовки.")
    if task == "segmentation":
        data = folder / "data.yaml"
        if not data.is_relative_to(folder) or not data.is_file():
            raise ValueError("Не найден подготовленный файл разметки внутри набора")
        for split in ("train", "val"):
            directory = folder / "images" / split
            if not directory.is_dir() or not any(directory.iterdir()):
                raise ValueError(f"Нет изображений в части {split}")
    else:
        old_root = Path(str(summary.get("output_dir") or folder))
        roots = []
        for value in summary.get("classification_roots", []):
            path = Path(value)
            if not path.is_absolute():
                roots.append((folder / path).resolve())
            else:
                try:
                    roots.append((folder / path.relative_to(old_root)).resolve())
                except ValueError:
                    raise ValueError("Каталог классификации находится вне подготовленного набора")
        if classification_root is None and len(roots) != 1:
            raise ValueError("Выберите один вариант классификации из подготовленного набора")
        data = classification_root.resolve() if classification_root else roots[0]
        if data not in roots or not data.is_relative_to(folder):
            raise ValueError("Выбран каталог вне подготовленного набора")
        for split in ("train", "val"):
            directory = data / split
            if not directory.is_dir() or not any(directory.rglob("*.*")):
                raise ValueError(f"Не подготовлена часть {split}; тестовую часть нельзя подменять валидационной")
    return {"task": "segment" if task == "segmentation" else "classify", "data": str(data),
            "summary": summary, "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--classification-root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error("Некорректные параметры обучения")
    if not args.model.is_file():
        parser.error("Нужен локальный файл исходной модели .pt")
    prepared = prepared_training_input(args.prepared, args.classification_root)
    install_offline_network_guard()
    import torch
    from ultralytics import YOLO
    configure_offline_ultralytics()
    if not torch.cuda.is_available():
        raise RuntimeError("Для обучения требуется CUDA PyTorch и видеокарта NVIDIA; проверьте окружение ИИ.")
    model = YOLO(str(args.model.resolve()))
    if model.task != prepared["task"]:
        kind = "сегментации" if prepared["task"] == "segment" else "классификации пород"
        raise ValueError(f"Для этого источника нужна исходная модель {kind}; выбранная модель имеет задачу {model.task}.")
    name = f"candidate_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    if prepared["task"] == "segment":
        import yaml
        config = yaml.safe_load(Path(prepared["data"]).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config.get("names"):
            raise ValueError("В подготовленном наборе нет справочника классов")
        config["path"] = str(args.prepared.resolve())
        config["train"], config["val"] = "images/train", "images/val"
        if (args.prepared / "images" / "test").is_dir():
            config["test"] = "images/test"
        else:
            config.pop("test", None)
        views = args.output / "dataset_views"
        views.mkdir(parents=True, exist_ok=True)
        current_data = views / (name + ".yaml")
        current_data.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        prepared["data"] = str(current_data.resolve())
    print("Обучение отдельной модели по исходным классам источника. Результат потребует проверки на ваших скважинах.", flush=True)
    model.train(data=prepared["data"], epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=0, workers=0, project=str(args.output.resolve()),
                name=name, exist_ok=False, patience=6, plots=False, amp=False, seed=42)
    run = Path(model.trainer.save_dir)
    best = run / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Обучение не сохранило best.pt")
    candidate = YOLO(str(best))
    validation = candidate.val(data=prepared["data"], split="val", device=0, workers=0,
                               batch=args.batch, imgsz=args.imgsz, plots=False, amp=False, project=str(run), name="validation")
    report = {
        "schema": "kern-analyzer-open-dataset-candidate-v1",
        "source_id": prepared["summary"].get("source_id"),
        "task": prepared["task"], "model_task": prepared["summary"].get("model_task"),
        "source_model": str(args.model.resolve()), "model": str(best),
        "prepared_dataset": str(args.prepared.resolve()),
        "dataset_summary_sha256": prepared["summary_sha256"],
        "license": prepared["summary"].get("license"),
        "classes": {str(key): str(value) for key, value in candidate.names.items()},
        "source_classes": prepared["summary"].get("classes"),
        "classification_subset": (str(Path(prepared["data"]).relative_to(args.prepared.resolve()))
                                  if prepared["task"] == "classify" else None),
        "split_notes": prepared["summary"].get("split_notes"),
        "validation_metrics": validation.results_dict,
        "status": "requires_validation_on_target_wells", "auto_activated": False,
        "offline_training": True, "amp": False,
        "note": "Source-domain validation is not accuracy on the project's 92 facies or complete geological descriptions.",
    }
    (run / "model_candidate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print("Кандидат сохранён: " + str(best), flush=True)
    print("Паспорт и проверка: " + str(run / "model_candidate.json"), flush=True)


if __name__ == "__main__":
    main()
