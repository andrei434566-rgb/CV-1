"""Qt transport for an offline, long-lived Qwen Python process."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from app.runtime_paths import bundled_root


class LocalVlmClient(QObject):
    reply = Signal(dict)
    state_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.readyReadStandardError.connect(self._read_error)
        self.process.finished.connect(self._finished)
        self._buffer = b""
        self._pending: set[str] = set()
        self._stderr_tail = ""
        self._python = ""
        self._slow_notice = QTimer(self)
        self._slow_notice.setSingleShot(True)
        self._slow_notice.timeout.connect(lambda: self.state_changed.emit("●  Подготавливаю ответ…"))
        self._request_timeout = QTimer(self)
        self._request_timeout.setSingleShot(True)
        self._request_timeout.timeout.connect(self._timed_out)

    def send(self, python: str, payload: dict) -> str:
        python_path = Path(python)
        runner = bundled_root() / "app" / "infrastructure" / "local_vlm_runner.py"
        if not python_path.is_file():
            raise FileNotFoundError("Не найден локальный Python для VLM. Укажите его в настройках ИИ.")
        if not runner.is_file():
            raise FileNotFoundError("Не найден локальный модуль VLM в приложении.")
        if self.process.state() != QProcess.ProcessState.NotRunning and self._python != python:
            self.stop()
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._python = python
            self._buffer = b""
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONIOENCODING", "utf-8")
            environment.insert("PYTHONUTF8", "1")
            self.process.setProcessEnvironment(environment)
            self.process.start(python, ["-u", str(runner)])
            if not self.process.waitForStarted(5000):
                raise RuntimeError("Не удалось запустить локальный Python VLM: " + self.process.errorString())
            self.state_changed.emit("●  Готов к работе")
        request_id = uuid4().hex
        payload = dict(payload, id=request_id)
        self._pending.add(request_id)
        self.process.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        self._slow_notice.start(8000)
        self._request_timeout.start(240000)
        return request_id

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            try:
                result = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            self._pending.discard(str(result.get("id")))
            if not self._pending:
                self._slow_notice.stop()
                self._request_timeout.stop()
            self.reply.emit(result)

    def _read_error(self) -> None:
        message = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_tail = (self._stderr_tail + message)[-2000:]

    def _finished(self, _code: int, _status) -> None:
        self._slow_notice.stop()
        self._request_timeout.stop()
        if self._pending:
            error = self._stderr_tail.strip() or "Локальный процесс VLM завершился до ответа."
            for request_id in tuple(self._pending):
                self.reply.emit({"id": request_id, "ok": False, "error": error})
            self._pending.clear()
        self.state_changed.emit("●  Помощник остановлен")

    def _timed_out(self) -> None:
        if not self._pending:
            return
        pending = tuple(self._pending)
        self._pending.clear()
        self._slow_notice.stop()
        self.process.kill()
        for request_id in pending:
            self.reply.emit({
                "id": request_id, "ok": False,
                "error": "Qwen не ответила за 4 минуты. Проверьте видеопамять и локальные пути модели в ⚙ → Настроить локальную VLM.",
            })

    def stop(self) -> None:
        self._slow_notice.stop()
        self._request_timeout.stop()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.closeWriteChannel()
            if self.process.waitForFinished(1500):
                return
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1000)
