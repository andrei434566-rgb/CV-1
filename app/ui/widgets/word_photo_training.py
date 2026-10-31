"""Simple Word + photograph training page; all file work runs off the UI thread."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QSettings, QThread, Signal, Qt, QSize
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.infrastructure.word_photo_training import (
    WordPhotoInspection, export_word_photo_pairs, inspect_word_photo_pairs,
)
from app.runtime_paths import user_data_root


class WordPhotoJob(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, *args, **kwargs):
        parent = kwargs.pop("parent", None)
        super().__init__(parent)
        self.operation, self.args, self.kwargs = operation, args, kwargs

    def run(self) -> None:
        try:
            self.succeeded.emit(self.operation(*self.args, **self.kwargs))
        except Exception as exc:
            self.failed.emit(str(exc))


class WordPhotoTrainingWidget(QWidget):
    train_requested = Signal()
    dataset_ready = Signal(str)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.inspection: WordPhotoInspection | None = None
        self.job: WordPhotoJob | None = None
        self._training_busy = False
        layout = QVBoxLayout(self)
        intro = QLabel(
            "1. Назовите фотографии 0001.jpg, 0002.jpg… и сложите в папку.\n"
            "2. В одном Word перед каждым описанием напишите «Фото 0001», «Фото 0002»…\n"
            "Текст из Excel можно вставлять абзацами или таблицей. Описания сохраняются целиком."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.photos = QLineEdit(str(settings.value("training/word_photos_folder", "") or ""))
        self.photos.setPlaceholderText("Папка с фотографиями")
        self.choose_photos = QPushButton("Папка фото…")
        self.choose_photos.clicked.connect(self._choose_photos)
        row = QHBoxLayout()
        row.addWidget(self.photos, 1)
        row.addWidget(self.choose_photos)
        layout.addLayout(row)
        self.document = QLineEdit(str(settings.value("training/word_document", "") or ""))
        self.document.setPlaceholderText("Один Word с описаниями (.docx)")
        self.choose_document = QPushButton("Word…")
        self.choose_document.clicked.connect(self._choose_document)
        self.choose_word_folder = QPushButton("Папка Word…")
        self.choose_word_folder.clicked.connect(self._choose_word_folder)
        row = QHBoxLayout()
        row.addWidget(self.document, 1)
        row.addWidget(self.choose_document)
        row.addWidget(self.choose_word_folder)
        layout.addLayout(row)
        self.check_button = QPushButton("Проверить пары")
        self.check_button.setObjectName("trainingPrimary")
        self.check_button.clicked.connect(self.check_pairs)
        layout.addWidget(self.check_button)
        self.status = QLabel("Выберите фотографии и Word, затем проверьте пары.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Фото", "Файл", "Проверка"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.currentCellChanged.connect(self._preview_pair)
        self.table.setMinimumHeight(85)
        self.table.setMaximumHeight(145)
        layout.addWidget(self.table)
        preview = QSplitter(Qt.Orientation.Horizontal)
        self.preview_image = QLabel("Фото выбранной пары")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumWidth(130)
        self.preview_image.setMaximumWidth(240)
        preview.addWidget(self.preview_image)
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("Здесь появится полный текст описания выбранного фото.")
        preview.addWidget(self.preview_text)
        preview.setMinimumHeight(90)
        preview.setMaximumHeight(125)
        layout.addWidget(preview, 1)
        self.approval = QCheckBox("Описания и соответствие номеров проверены специалистом")
        self.approval.toggled.connect(self._refresh_buttons)
        layout.addWidget(self.approval)
        self.export_button = QPushButton("Подготовить проверенные пары")
        self.export_button.setObjectName("trainingPrimary")
        self.export_button.clicked.connect(self.export_pairs)
        layout.addWidget(self.export_button)
        self.manifest_label = QLabel()
        self.manifest_label.setWordWrap(True)
        layout.addWidget(self.manifest_label)
        self.continue_adapter = QCheckBox("Продолжить с подключённой дообученной модели")
        layout.addWidget(self.continue_adapter)
        self.train_button = QPushButton("Начать локальное обучение на этих парах")
        self.train_button.setObjectName("trainingPrimary")
        self.train_button.clicked.connect(self.train_requested)
        layout.addWidget(self.train_button)
        note = QLabel(
            "Проверка и подготовка работают без видеокарты. Обучение использует локальную Qwen и NVIDIA. "
            "Новая версия сохраняется отдельно. Для обучения границ сегментации нужны проверенные контуры на фото."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.photos.textChanged.connect(self._invalidate)
        self.document.textChanged.connect(self._invalidate)
        self._refresh_buttons()

    @property
    def busy(self) -> bool:
        return self.job is not None

    def _choose_photos(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка фотографий для обучения", self.photos.text())
        if folder:
            self.photos.setText(folder)

    def _choose_document(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Word с описаниями по номерам фото", self.document.text(), "Word (*.docx)")
        if filename:
            self.document.setText(filename)

    def _choose_word_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка с Word-описанием")
        if not folder:
            return
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.casefold() == ".docx" and not p.name.startswith("~$"))
        if not files:
            self.status.setText("В выбранной папке нет Word (.docx).")
        elif len(files) == 1:
            self.document.setText(str(files[0]))
        else:
            name, accepted = QInputDialog.getItem(self, "Word с описаниями", "Выберите один документ:", [p.name for p in files], 0, False)
            if accepted:
                self.document.setText(str(Path(folder) / name))

    def _invalidate(self) -> None:
        self.inspection = None
        self.approval.setChecked(False)
        self.table.setRowCount(0)
        self.preview_image.clear()
        self.preview_text.clear()
        self.settings.remove("training/word_pairs_manifest")
        self.status.setText("Пути изменены. Нажмите «Проверить пары».")
        self._refresh_buttons()

    def check_pairs(self) -> None:
        if self.busy or self._training_busy:
            return
        self._invalidate()
        self.settings.setValue("training/word_photos_folder", self.photos.text().strip())
        self.settings.setValue("training/word_document", self.document.text().strip())
        self._start_job(inspect_word_photo_pairs, self.photos.text().strip(), self.document.text().strip())
        self.status.setText("Читаю Word, проверяю номера и фотографии…")

    def export_pairs(self) -> None:
        if self.busy or self._training_busy or not self.inspection or not self.approval.isChecked():
            return
        destination = user_data_root() / "vlm_training" / "word_photo" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        self._start_job(export_word_photo_pairs, self.inspection, destination, approved=True)
        self.status.setText("Копирую проверенные пары в локальную обучающую базу…")

    def _start_job(self, operation, *args, **kwargs) -> None:
        self.job = WordPhotoJob(operation, *args, parent=self, **kwargs)
        self.job.succeeded.connect(self._job_succeeded)
        self.job.failed.connect(self._job_failed)
        self.job.finished.connect(self._job_finished)
        self.job.finished.connect(self.job.deleteLater)
        self._refresh_buttons()
        self.job.start()

    def _job_succeeded(self, result) -> None:
        if isinstance(result, WordPhotoInspection):
            self.inspection = result
            self.table.setRowCount(len(result.pairs))
            for index, pair in enumerate(result.pairs):
                for column, text in enumerate((pair.photo_id, ", ".join(p.name for p in pair.images), pair.message)):
                    item = QTableWidgetItem(text)
                    item.setToolTip(text)
                    self.table.setItem(index, column, item)
            count = len(result.ready_pairs)
            skipped = len(result.pairs) - count
            self.status.setText(f"Готово пар: {count}. С ошибками или без описания: {skipped}. В обучение попадут только готовые пары.")
            if result.pairs:
                self.table.selectRow(0)
                self._preview_pair(0)
        else:
            self.settings.setValue("training/word_pairs_manifest", str(result["manifest"]))
            self.settings.sync()
            self.status.setText(f"Подготовлено пар: {result['count']}. Пропущено: {result['skipped']}. Можно начать обучение.")
            self.dataset_ready.emit(str(result["manifest"]))

    def _job_failed(self, message: str) -> None:
        self.status.setText("Не удалось подготовить пары: " + message)

    def _job_finished(self) -> None:
        self.job = None
        self._refresh_buttons()

    def _preview_pair(self, row: int, *_args) -> None:
        if not self.inspection or not 0 <= row < len(self.inspection.pairs):
            return
        pair = self.inspection.pairs[row]
        self.preview_text.setPlainText(pair.description)
        self.preview_image.clear()
        if len(pair.images) == 1:
            reader = QImageReader(str(pair.images[0]))
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                reader.setScaledSize(size.scaled(QSize(220, 200), Qt.AspectRatioMode.KeepAspectRatio))
            image = reader.read()
            if not image.isNull():
                self.preview_image.setPixmap(QPixmap.fromImage(image))
                return
        self.preview_image.setText(pair.message)

    def set_training_busy(self, value: bool) -> None:
        self._training_busy = value
        self._refresh_buttons()

    def _refresh_buttons(self, *_args) -> None:
        busy = self.busy or self._training_busy
        for control in (self.photos, self.document, self.choose_photos, self.choose_document, self.choose_word_folder,
                        self.check_button, self.approval, self.continue_adapter):
            control.setEnabled(not busy)
        self.export_button.setEnabled(not busy and bool(self.inspection and self.inspection.ready_pairs) and self.approval.isChecked())
        manifest = str(self.settings.value("training/word_pairs_manifest", "") or "")
        self.train_button.setEnabled(not busy and Path(manifest).is_file())
        self.manifest_label.setText("Подготовленная база: " + (manifest if Path(manifest).is_file() else "пока не создана"))
