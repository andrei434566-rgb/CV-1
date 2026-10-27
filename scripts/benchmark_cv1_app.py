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
parser.add_argument("--output", type=Path, default=Path("outputs/app_pipeline_benchmark"))
args = parser.parse_args()
if len(set(path.resolve() for path in args.photos)) != len(args.photos):
    parser.error("Use distinct source photographs, not repeated cached inputs")
if any(not path.is_file() for path in args.photos) or args.model is not None and not args.model.is_file():
    parser.error("An input photo or model does not exist")
args.output.mkdir(parents=True, exist_ok=True)
application = QApplication([])
application.setQuitOnLastWindowClosed(False)
settings = QSettings(str(args.output / "isolated_settings.ini"), QSettings.Format.IniFormat)
settings.clear()
with patch("app.ui.windows.main_window.QSettings", return_value=settings):
    window = MainWindow()
window._resolve_model_path = lambda: args.model
window._project_folder = None
state = {"started": None, "chat_available": False, "errors": []}


def segmentation_error(message):
    window._segmentation_failed = True
    state["errors"].append(message)


window._segmentation_error = segmentation_error


def begin():
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
                  "route": "real MainWindow + QProcess + local runtime; offscreen Qt", "response_cache": False,
                  "all_hypotheses_are_unconfirmed": unconfirmed}
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        window.close()
        application.exit(0 if success else 1)


timer = QTimer()
timer.setInterval(50)
timer.timeout.connect(check)
timer.start()
QTimer.singleShot(0, begin)
raise SystemExit(application.exec())
