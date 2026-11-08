"""Explicit central training creates candidates, never switches production weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.infrastructure.model_hub import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description="CV-1: обучение кандидата на принятых серверных данных")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kind", choices=["geometry"], required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2)
    args = parser.parse_args()
    report = read_json(args.dataset / "dataset_report.json")
    if not report.get("independent_validation_available"):
        raise ValueError("Для центрального цикла нужны раздельные группы обучения и проверки")
    if args.output.exists():
        raise ValueError("Выберите новую папку кандидата. Существующие веса не перезаписываются")
    if not args.base.is_file(): raise ValueError("Не найдены локальные исходные веса YOLO")
    from ultralytics import YOLO
    import torch
    model = YOLO(str(args.base))
    result = model.train(data=str(args.dataset / "yolo/data.yaml"), epochs=max(1, int(args.epochs)),
        imgsz=640, device=0 if torch.cuda.is_available() else "cpu", amp=False, workers=0,
        project=str(args.output.parent), name=args.output.name, exist_ok=False, plots=False)
    metrics = {key: float(value) for key, value in result.results_dict.items()}
    candidate = {"accepted": False, "independent_wells": report.get("validation_groups", []),
        "candidate": str(args.output), "metrics": metrics, "baseline_metrics": None,
        "note": "Кандидат не опубликован. Проверьте отдельные скважины, сравните с предыдущей версией и оформите решение администратора."}
    atomic_json(args.output / "validation_pending.json", candidate)
    print(json.dumps(candidate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
