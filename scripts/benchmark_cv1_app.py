"""Exercise the real MainWindow/QProcess pipeline in an isolated offscreen project."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtWidgets import QApplication
from app.ui.windows.main_window import MainWindow
from app.infrastructure.cv1_analysis import TEXT_FIELDS

parser = argparse.ArgumentParser()
parser.add_argument("photos", type=Path, nargs="+")
parser.add_argument("--model", type=Path)
parser.add_argument("--adapter", type=Path, help="Explicit candidate LoRA for an isolated comparison; does not change user settings")
parser.add_argument("--prepare", action="store_true", help="Prepare CPU imports, as at normal application startup")
parser.add_argument("--output", type=Path, default=Path("outputs/app_pipeline_benchmark"))
args = parser.parse_args()
if len(set(path.resolve() for path in args.photos)) != len(args.photos):
    parser.error("Use distinct source photographs, not repeated cached inputs")
if any(not path.is_file() for path in args.photos) or args.model is not None and not args.model.is_file():
    parser.error("An input photo or model does not exist")
if args.adapter is not None and not (args.adapter / "adapter_model.safetensors").is_file():
    parser.error("The candidate adapter does not exist")
args.output.mkdir(parents=True, exist_ok=True)
application = QApplication([])
application.setQuitOnLastWindowClosed(False)
settings = QSettings(str(args.output / "isolated_settings.ini"), QSettings.Format.IniFormat)
settings.clear()
if args.adapter is not None:
    settings.setValue("vlm/model_mode", "lora")
    settings.setValue("vlm/adapter", str(args.adapter.resolve()))
with patch("app.ui.windows.main_window.QSettings", return_value=settings):
    window = MainWindow()
window._resolve_model_path = lambda: args.model
window._project_folder = None
state = {"started": None, "chat_available": False, "errors": [], "worker_replies": []}
state["preparation_started"] = perf_counter()
state["preparation_seconds"] = 0.0
state["photo_finished"] = {}
original_interval_reply = window._on_cv1_interval_reply


def interval_reply(result):
    state["worker_replies"].append(result)
    original_interval_reply(result)
    for record in window._records:
        if (record.identifier not in state["photo_finished"] and record.detections
                and all(row.attributes.get("__interpretation_status") in {"draft", "error"} for row in record.detections)):
            state["photo_finished"][record.identifier] = perf_counter() - state["started"]


window._on_cv1_interval_reply = interval_reply


def segmentation_error(message):
    window._segmentation_failed = True
    state["errors"].append(message)


window._segmentation_error = segmentation_error


def begin():
    if window._runtime_preparation_id:
        QTimer.singleShot(50, begin)
        return
    state["preparation_seconds"] = perf_counter() - state["preparation_started"] if args.prepare else 0.0
    state["started"] = perf_counter()
    window._import_chat_analysis_photos(args.photos)


def check():
    if state["started"] is None:
        return
    elapsed = perf_counter() - state["started"]
    if window._analysis_active:
        state["chat_available"] |= bool(window.assistant_panel.send_button.isEnabled()
                                        and window._activity_dialog.windowModality() == Qt.WindowModality.NonModal)
    idle = not window._analysis_active and window._analysis_pending is None and window._segmentation_thread is None
    if (idle and len(window._records) == len(args.photos)) or elapsed > 180:
        timer.stop()
        rows = [layer for record in window._records for layer in record.detections]
        complete = sum(row.attributes.get("__interpretation_status") == "draft"
                       and all(str(row.attributes.get(names[0]) or "").strip() for names in TEXT_FIELDS.values())
                       for row in rows)
        unconfirmed = not any(row.training_ready for row in rows)
        success = (bool(rows) and complete == len(rows) and len(window._records) == len(args.photos)
                   and all(record.core_columns for record in window._records)
                   and unconfirmed and not state["errors"])
        report = {"photos": len(window._records), "objects": len(rows), "complete_descriptions": complete,
                  "total_seconds": elapsed, "chat_available_during_processing": state["chat_available"],
                  "segmentation_model": str(args.model) if args.model else None, "errors": state["errors"],
                  "candidate_adapter": str(args.adapter) if args.adapter else None,
                  "route": "real MainWindow + QProcess + local runtime; offscreen Qt", "response_cache": False,
                  "all_hypotheses_are_unconfirmed": unconfirmed}
        report["startup_cpu_import_seconds"] = state["preparation_seconds"]
        (args.output / "worker_replies.json").write_text(
            json.dumps(state["worker_replies"], ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "worker_stderr.txt").write_text(window._vlm_client._stderr_tail, encoding="utf-8")
        report["per_photo"] = []
        previous_finish = 0.0
        for record in window._records:
            finished = state["photo_finished"].get(record.identifier)
            report["per_photo"].append({"photo": record.path, "objects": len(record.detections),
                "seconds_until_complete": finished,
                "seconds_after_previous_photo": finished - previous_finish if finished is not None else None,
                "descriptions": [dict(row.attributes) for row in record.detections]})
            if finished is not None:
                previous_finish = finished
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        compact = {key: value for key, value in report.items() if key != "per_photo"}
        compact["per_photo_seconds"] = [row["seconds_after_previous_photo"] for row in report["per_photo"]]
        print(json.dumps(compact, ensure_ascii=True), flush=True)
        window.close()
        application.exit(0 if success else 1)


timer = QTimer()
timer.setInterval(50)
timer.timeout.connect(check)
timer.start()
QTimer.singleShot(0, begin)
if args.prepare:
    window.prepare_local_runtime()
raise SystemExit(application.exec())
