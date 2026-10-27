"""Uncached end-to-end core geometry, assembly and full VLM description benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from app.domain.models import PhotoRecord
from app.infrastructure.image_loading import load_working_pixmap
from app.infrastructure.cv1_analysis import (PIPELINE_MODE, IntervalJob, apply_interval_description,
                                           create_core_column_description_detections, geometry_signature, write_interval_crop)
from app.infrastructure.ml.core_column_service import CoreColumnRecognizer, assemble_core_tape, read_bgr_image
from app.infrastructure.local_vlm_runner import LocalModel

parser = argparse.ArgumentParser()
parser.add_argument("photos", type=Path, nargs="+")
parser.add_argument("--base", type=Path)
parser.add_argument("--model", type=Path, help="An optional actual YOLO segmentation checkpoint")
parser.add_argument("--output", type=Path, default=Path("outputs/cv1_pipeline_benchmark"))
args = parser.parse_args()
if len(set(path.resolve() for path in args.photos)) != len(args.photos):
    parser.error("Use distinct source photographs; repeated/cache-hit inputs are not a throughput benchmark.")
if args.base is None:
    snapshots = Path(__file__).resolve().parents[2] / "vlm_core_v1/model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots"
    args.base = next(path for path in snapshots.iterdir() if (path / "config.json").is_file())
args.output.mkdir(parents=True, exist_ok=True)
application = QApplication.instance() or QApplication([])
started = perf_counter()
recognizer, model = CoreColumnRecognizer(), LocalModel()
segmentation = None
if args.model is not None:
    from app.infrastructure.ml.yolo_model_service import YoloModelService
    segmentation = YoloModelService(args.model, image_size=640, geometry_only=True)
reports, prepared, successes, total = [], [], 0, 0
try:
    for photo_number, photo in enumerate(args.photos, 1):
        photo_started = perf_counter()
        record = PhotoRecord(uuid4().hex, str(photo), load_working_pixmap(photo), analysis_mode=PIPELINE_MODE)
        record.core_columns = recognizer.recognize_path(photo, (record.pixmap.width(), record.pixmap.height()))
        if not record.core_columns:
            raise RuntimeError(f"No core columns found in {photo.name}; do not count an empty pass as success")
        if segmentation is not None:
            record.detections = segmentation.predict(str(photo), (record.pixmap.width(), record.pixmap.height()), record.core_columns)
        record.detections.extend(create_core_column_description_detections(record))
        source = read_bgr_image(photo)
        sx, sy = source.shape[1] / record.pixmap.width(), source.shape[0] / record.pixmap.height()
        boxes = [{key: value * (sx if key in {"left", "right"} else sy) for key, value in box.items()} for box in record.core_columns]
        tape = assemble_core_tape(source, boxes)
        import cv2
        ok, encoded = cv2.imencode(".jpg", tape.image)
        if not ok:
            raise RuntimeError("Assembly export failed")
        (args.output / f"tape_{photo_number:02d}.jpg").write_bytes(encoded.tobytes())
        geometry_seconds = perf_counter() - photo_started
        jobs = [IntervalJob.create(record, detection) for detection in record.detections]
        reports.append({"photo": str(photo.resolve()), "columns": len(record.core_columns), "objects": len(jobs),
                        "geometry_assembly_seconds": geometry_seconds, "results": []})
        prepared.extend((record, job, len(reports) - 1) for job in jobs)
    # The app completes its geometry stage before starting the shared VLM.
    if segmentation is not None:
        del segmentation
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    for offset in range(0, len(prepared), 4):
        selected = prepared[offset:offset + 4]
        items, crops = [], []
        for index, (record, job, _) in enumerate(selected, 1):
            crop = args.output / f"crop_{offset + index}.png"
            write_interval_crop(record, job, crop)
            crops.append(crop)
            items.append({"id": str(index), "image": str(crop), "whole_column": job.detection.attributes.get("__core_column_description") == "1",
                          "context": "Описание отдельного найденного столбика или полигона керна."})
        response = model.answer({"operation": "describe_batch", "base": str(args.base), "model_mode": "base", "items": items}, [])
        for index, ((record, job, report_index), crop) in enumerate(zip(selected, crops), 1):
            reply = response["items"][str(index)]
            try:
                apply_interval_description(record, job, reply)
                if geometry_signature(job.detection) != job.geometry_sha256 or job.detection.training_ready:
                    raise RuntimeError("Geometry or review contract changed")
                successes += 1
            except ValueError as exc:
                reply["validation_error"] = str(exc)
            reports[report_index]["results"].append(reply)
            total += 1
            crop.unlink(missing_ok=True)
        print(json.dumps({"described_objects": total, "complete_descriptions": successes,
                          "elapsed_seconds": perf_counter() - started}, ensure_ascii=False), flush=True)
finally:
    model.turbo.close()
summary = {"photos": len(reports), "unique_photos": len(reports), "objects": total, "complete_descriptions": successes,
           "total_seconds": perf_counter() - started, "segmentation_model": str(args.model) if args.model else None,
           "response_cache": False, "results": reports}
(args.output / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False), flush=True)
if successes != total or not total:
    raise SystemExit(1)
