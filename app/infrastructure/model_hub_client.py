"""Qt orchestration; a blocked SMB read cannot freeze or prevent GUI shutdown."""
from pathlib import Path
import sys
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from app.infrastructure.model_hub import atomic_json, read_json
from app.runtime_paths import bundled_root, user_data_root, bundled_runtime_defaults


class ModelHubClient(QObject):
    finished = Signal(str, dict)
    status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.finished.connect(self._finished)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timed_out)
        self.operation, self.result = "", None

    @property
    def busy(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, operation: str, hub: str, snapshot: dict | None = None):
        if self.busy:
            return False
        root = user_data_root() / "network_cache"
        job = root / "jobs" / (uuid4().hex + ".json")
        self.result = job.with_suffix(".result.json")
        atomic_json(job, {"operation": operation, "hub": hub, "cache": str(root), "outbox": str(root / "outbox"),
                         "snapshot": snapshot, "result": str(self.result)})
        self.operation = operation
        portable_python = bundled_runtime_defaults().get('python', '')
        script = bundled_root() / "app/infrastructure/model_hub_worker.py"
        if portable_python:
            # Own the worker itself: stopping QProcess must stop SMB writes too.
            self.process.start(portable_python, ["-B", str(script), str(job)])
        elif getattr(sys, "frozen", False):
            self.process.start(sys.executable, ["--hub-sync-job", str(job)])
        else:
            self.process.start(sys.executable, ["-B", str(script), str(job)])
        if not self.process.waitForStarted(2000):
            self.finished.emit(operation, {"ok": False, "error": self.process.errorString()})
            return False
        self.timer.start(20 * 60 * 1000)
        self.status_changed.emit("Проверка общих моделей…" if operation == "sync" else "Передача проверенных данных…")
        return True

    def _finished(self, _code, _status):
        self.timer.stop()
        try:
            result = read_json(self.result) if self.result and self.result.is_file() else {"ok": False, "error": "Операция прервана; локальные модели сохранены"}
        except (OSError, ValueError) as exc:
            result = {"ok": False, "error": str(exc)}
        self.finished.emit(self.operation, result)

    def _timed_out(self):
        self.process.kill()
        self.status_changed.emit("Общее хранилище не ответило; используются локальные модели")

    def stop(self):
        self.timer.stop()
        if self.busy:
            self.process.kill()
            self.process.waitForFinished(1500)
