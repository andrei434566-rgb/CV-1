"""One optional window for downloading and preparing public core datasets."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QInputDialog,
)

from app.infrastructure.open_dataset_library import (
    available_files, last_preparation, prepare_source, read_state, register_files, write_state,
)
from app.infrastructure.open_dataset_sources import download_artifact, get_source, list_sources, resolve_artifacts, verify_artifact
from app.runtime_paths import application_root, bundled_root, user_data_root


class DatasetJob(QThread):
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, operation: str, source_id: str, root: Path, app_root: Path, parent=None, *, config=None):
        super().__init__(parent)
        self.operation, self.source_id = operation, source_id
        self.root, self.app_root = root, app_root
        self.cancellation = threading.Event()
        self.config = config

    def cancel(self) -> None:
        self.cancellation.set()

    def run(self) -> None:
        try:
            if self.operation == "training_runtime":
                from app.infrastructure.cv_training_runtime import resolve_cv_training_python
                python = resolve_cv_training_python(self.config, root=self.app_root, cancel=self.cancellation)
                if self.cancellation.is_set():
                    raise InterruptedError("Проверка окружения отменена")
                self.completed.emit({"operation": self.operation, "python": str(python)})
            elif self.operation == "download":
                artifacts = resolve_artifacts(self.source_id, cancel=self.cancellation)
                if not artifacts:
                    raise ValueError("В источнике не найдены файлы. Можно загрузить архив вручную и выбрать его здесь.")
                paths = []
                folder = self.root / self.source_id / "archives"
                local = available_files(self.root, self.source_id, self.app_root)
                for artifact in artifacts:
                    if self.cancellation.is_set():
                        raise InterruptedError("Загрузка отменена; её можно продолжить позже")
                    path = next((path for path in local if path.name == artifact.filename
                                 and verify_artifact(path, artifact, cancel=self.cancellation)), None)
                    if path is None:
                        path = download_artifact(artifact, folder, cancel=self.cancellation, progress=self.progress.emit)
                    else:
                        self.progress.emit({"message": "Использую уже проверенный файл: " + path.name})
                    paths.append(path)
                    register_files(self.root, self.source_id, [path])
                self.completed.emit({"operation": "download", "source_id": self.source_id,
                                     "files": [str(path) for path in paths]})
            else:
                result = prepare_source(
                    self.root, self.source_id, available_files(self.root, self.source_id, self.app_root),
                    progress=self.progress.emit, cancel=self.cancellation,
                )
                self.completed.emit({**result, "operation": "prepare"})
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)


class OpenDatasetsDialog(QDialog):
    materials_ready = Signal(dict)

    def __init__(self, main_window, *, source_id: str | None = None, files: list[Path] | None = None):
        super().__init__(main_window)
        self.main = main_window
        self.root = user_data_root() / "open_datasets"
        self._job: DatasetJob | None = None
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._training_output)
        self._process.finished.connect(self._training_finished)
        self._training_source = ""
        self._training_cancelled = False
        self._pending_training = None
        self._sources = list_sources()
        self.setWindowTitle("Учебные материалы из открытых источников")
        self.resize(940, 680)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Выберите источник, скачайте или добавьте архив и подготовьте материалы. "
            "Фото и таблицы хранятся на этом компьютере. Интернет нужен только для загрузки."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(len(self._sources), 4)
        self.table.setHorizontalHeaderLabels(["Источник", "Назначение", "Условия", "На компьютере"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setMinimumHeight(85)
        layout.addWidget(self.details)
        buttons = QHBoxLayout()
        self.download_button = QPushButton("Скачать")
        self.download_button.clicked.connect(lambda: self._start("download"))
        buttons.addWidget(self.download_button)
        self.local_button = QPushButton("Выбрать архивы на ПК…")
        self.local_button.clicked.connect(self._choose_files)
        buttons.addWidget(self.local_button)
        self.prepare_button = QPushButton("Подготовить материалы")
        self.prepare_button.clicked.connect(lambda: self._start("prepare"))
        buttons.addWidget(self.prepare_button)
        self.review_button = QPushButton("Открыть результат")
        self.review_button.clicked.connect(self._open_result)
        buttons.addWidget(self.review_button)
        self.train_button = QPushButton("Обучить модель…")
        self.train_button.clicked.connect(self._train)
        buttons.addWidget(self.train_button)
        layout.addLayout(buttons)
        self.status = QLabel("Готово к работе")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(125)
        layout.addWidget(self.log)
        bottom = QHBoxLayout()
        source_button = QPushButton("Страница источника")
        source_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.selected_source().primary_url)))
        bottom.addWidget(source_button)
        bottom.addStretch(1)
        self.cancel_button = QPushButton("Отменить операцию")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        bottom.addWidget(self.cancel_button)
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.reject)
        bottom.addWidget(close_button)
        layout.addLayout(bottom)
        self.refresh()
        selected = next((i for i, item in enumerate(self._sources) if item.id == source_id), 0)
        self.table.selectRow(selected)
        if files:
            register_files(self.root, self.selected_source().id, files)
            self.refresh()
        self._selection_changed()

    def selected_source(self):
        return self._sources[max(0, self.table.currentRow())]

    def refresh(self) -> None:
        for row, source in enumerate(self._sources):
            paths = available_files(self.root, source.id, application_root())
            prepared = last_preparation(self.root, source.id)
            state = "Подготовлены" if prepared else f"Файлов: {len(paths)}" if paths else "Не загружен"
            if prepared and prepared.get("counts", {}).get("paired_candidates") is not None:
                state = "Готовы к проверке"
            roles = {"columns": "Столбики", "facies": "Фации", "description": "Описания",
                     "intervals": "Интервалы", "lithology": "Породы"}
            values = (source.name_ru, ", ".join(roles.get(role, role) for role in source.roles), source.license_id, state)
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        self._selection_changed()

    def _selection_changed(self) -> None:
        source = self.selected_source()
        note = "Исследовательский набор: лицензия ограничивает коммерческое использование.\n" if source.noncommercial else ""
        self.details.setText(source.name_ru + "\n" + note + str(source.notes))
        busy = self._job is not None or self._process.state() != QProcess.ProcessState.NotRunning
        prepared = last_preparation(self.root, source.id)
        for button in (self.download_button, self.local_button):
            button.setEnabled(not busy)
        self.prepare_button.setEnabled(not busy and bool(available_files(self.root, source.id, application_root())))
        self.review_button.setEnabled(not busy and prepared is not None)
        trainable = prepared is not None and any(
            (Path(str(prepared.get("output_dir"))) / name).is_file() for name in ("data.yaml", "summary.json")
        ) and prepared.get("readiness") == "ready" and source.id not in {"ukcs", "iodp362"}
        self.train_button.setEnabled(not busy and trainable)
        self.cancel_button.setEnabled(busy and not self._training_cancelled)
        self.table.setEnabled(not busy)

    def _choose_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(
            self, "Архивы или документы источника", "",
            "Архивы и документы (*.zip *.xlsx *.csv *.tsv *.pdf *.docx *.txt *.json);;Все файлы (*)",
        )
        if names:
            register_files(self.root, self.selected_source().id, [Path(name) for name in names])
            self.refresh()

    def _start(self, operation: str) -> None:
        if self._job is not None:
            return
        source = self.selected_source()
        job = DatasetJob(operation, source.id, self.root, application_root(), self)
        self._job = job
        job.progress.connect(self._progress)
        job.completed.connect(self._completed)
        job.failed.connect(self._failed)
        job.finished.connect(self._job_finished)
        self.progress.setRange(0, 0)
        self.status.setText("Проверяю источник…" if operation == "download" else "Подготавливаю материалы…")
        self._selection_changed()
        job.start()

    def _progress(self, event: dict) -> None:
        total = int(event.get("total_bytes") or event.get("total") or 0)
        done = int(event.get("downloaded_bytes") or event.get("completed") or 0)
        if total:
            self.progress.setRange(0, 100)
            self.progress.setValue(min(100, round(done * 100 / total)))
        else:
            self.progress.setRange(0, 0)
        text = str(event.get("message") or event.get("filename") or event.get("stage") or "Подготовка")
        if event.get("total_bytes"):
            text += f" · {done / 1024**2:.1f} / {total / 1024**2:.1f} МБ"
        self.status.setText(text)

    def _completed(self, result: dict) -> None:
        if result["operation"] == "training_runtime":
            self._launch_training(Path(result["python"]))
            return
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText("Файлы скачаны. Нажмите «Подготовить материалы»." if result["operation"] == "download"
                            else "Материалы подготовлены. Откройте результат для проверки.")
        if result["operation"] == "download":
            self.log.append("Скачаны и проверены файлы:\n" + "\n".join(Path(value).name for value in result.get("files", [])))
        else:
            counts = result.get("counts", {})
            self.log.append("Результат: " + str(result.get("output_dir", "")) + "\n"
                            + "Количество записей: " + json.dumps(counts, ensure_ascii=False))
        if result["operation"] == "prepare":
            self.materials_ready.emit(result)

    def _failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText(message)
        self.log.append(message)

    def _job_finished(self) -> None:
        job, self._job = self._job, None
        self._pending_training = None
        if job is not None:
            job.deleteLater()
        self.refresh()

    def _cancel(self) -> None:
        if self._job:
            self._job.cancel()
            self.status.setText("Останавливаю операцию…")
            self.cancel_button.setEnabled(False)
        elif self._process.state() != QProcess.ProcessState.NotRunning:
            self._training_cancelled = True
            self.status.setText("Останавливаю обучение. Уже сохранённые файлы останутся на компьютере…")
            self.cancel_button.setEnabled(False)
            self._process.terminate()
            QTimer.singleShot(3000, self._kill_cancelled_training)

    def _kill_cancelled_training(self) -> None:
        if self._training_cancelled and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    def _open_result(self) -> None:
        result = last_preparation(self.root, self.selected_source().id)
        if not result:
            return
        if result.get("counts", {}).get("paired_candidates", 0):
            from app.ui.dialogs.open_dataset_review_dialog import OpenDatasetReviewDialog
            dialog = OpenDatasetReviewDialog(result, self)
            dialog.pairs_ready.connect(self._reviewed_pairs_ready)
            dialog.exec()
            return
        path = Path(str(result.get("review_html") or result.get("output_dir")))
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _reviewed_pairs_ready(self, result: dict) -> None:
        settings = self.main._settings
        settings.setValue("training/pairs_manifest", result["manifest"])
        settings.setValue("training/pairs_holdout_manifest", result["holdout_manifest"])
        settings.remove("training/pairs_additional_manifest")
        settings.sync()
        self.status.setText(
            f"Проверенные пары готовы: обучение {result['counts']['train']}, "
            f"контроль {result['counts']['holdout']}. Они подключены к обучению описаний."
        )
        self.materials_ready.emit({**result, "source_id": self.selected_source().id,
                                   "operation": "reviewed_pairs"})

    def _train(self) -> None:
        result = last_preparation(self.root, self.selected_source().id)
        if not result:
            return
        config = self.main._local_vlm_config()
        weight, _ = QFileDialog.getOpenFileName(
            self, "Исходная модель для дообучения", "", "Обучаемые модели (*.pt)"
        )
        if not weight:
            return
        if self.main._vlm_pending is not None:
            QMessageBox.information(self, "Обучение", "Дождитесь текущего ответа ассистента.")
            return
        source_id = self.selected_source().id
        extra_args = []
        roots = result.get("classification_roots") or []
        if len(roots) > 1:
            labels = [Path(str(value)).name for value in roots]
            selected, accepted = QInputDialog.getItem(
                self, "Вариант набора", "Какой набор классов использовать?",
                labels, 0, False,
            )
            if not accepted:
                return
            extra_args = ["--classification-root", str(roots[labels.index(selected)])]
        self.main._vlm_client.stop()
        self._pending_training = (source_id, result, weight, extra_args)
        job = DatasetJob("training_runtime", source_id, self.root, application_root(), self, config=config)
        self._job = job
        job.completed.connect(self._completed)
        job.failed.connect(self._failed)
        job.finished.connect(self._job_finished)
        self.status.setText("Проверяю локальное окружение для обучения…")
        self.progress.setRange(0, 0)
        self._selection_changed()
        job.start()

    def _launch_training(self, python: Path) -> None:
        if not self._pending_training or (self._job and self._job.cancellation.is_set()):
            return
        source_id, result, weight, extra_args = self._pending_training
        if self.main._vlm_pending is not None:
            self._failed("Дождитесь текущего ответа ассистента, затем запустите обучение.")
            return
        self.main._vlm_client.stop()
        script = bundled_root() / "app" / "infrastructure" / "ml" / "open_dataset_train.py"
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("HF_HUB_OFFLINE", "1")
        environment.insert("TRANSFORMERS_OFFLINE", "1")
        environment.insert("YOLO_OFFLINE", "true")
        environment.insert("YOLO_AUTOINSTALL", "false")
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self._process.setProcessEnvironment(environment)
        self._training_cancelled = False
        self._process.start(str(python), ["-u", str(script), "--prepared", str(result["output_dir"]),
                                          "--model", weight, "--output", str(self.root / source_id / "models"), *extra_args])
        self._training_source = source_id
        if not self._process.waitForStarted(3000):
            self._failed(self._process.errorString())
            return
        self.status.setText("Обучение отдельной модели. Результат будет сохранён для проверки.")
        self.progress.setRange(0, 0)
        self._selection_changed()

    def _training_output(self) -> None:
        value = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if value.strip():
            self.log.append(value.strip()[-2200:])

    def _training_finished(self, code: int, _status) -> None:
        self._training_output()
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if code == 0 else 0)
        self.status.setText("Обучение остановлено. Рабочая модель не заменялась." if self._training_cancelled else
                            "Обучение завершено. Модель сохранена как кандидат; подключение после проверки."
                            if code == 0 else "Обучение не завершилось. Причина показана в журнале.")
        self._training_cancelled = False
        self.refresh()

    def reject(self) -> None:
        if self._job is not None:
            self._cancel()
            return
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self.status.setText("Идёт обучение. Закройте окно после его завершения.")
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._job is not None or self._process.state() != QProcess.ProcessState.NotRunning:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
