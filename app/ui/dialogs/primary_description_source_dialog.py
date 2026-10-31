"""Select persistent, offline reference documents for facies and flow regime."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout

from app.infrastructure.primary_description_source import import_primary_sources, source_index
from app.runtime_paths import user_data_root
from app.ui.widgets.word_photo_training import WordPhotoJob


class PrimaryDescriptionSourceDialog(QDialog):
    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.folder = current
        self.worker = None
        self.import_result = None
        self.setWindowTitle("Основной источник · фации и гидродинамический режим")
        self.resize(740, 510)
        self.setStyleSheet("""
            QWidget { background: #f6faff; color: #153e77; font: 13px "Segoe UI"; }
            QLabel { background: transparent; }
            QListWidget { background: white; border: 1px solid #cfdfef; border-radius: 7px; padding: 8px; }
            QPushButton { background: #edf5ff; color: #0869c4; border: 1px solid #c8e0f8; border-radius: 7px; padding: 8px 12px; }
            QPushButton:disabled { color: #8a9aaf; }
            QPushButton#primarySourceApply { background: #087ce8; color: white; }
        """)
        layout = QVBoxLayout(self)
        intro = QLabel("Выберите проверенные материалы специалиста. Они будут основным справочником "
                       "именно для фаций и гидродинамического режима. Цвет, порода и текстура по-прежнему "
                       "описываются по фото. Правки специалиста для конкретного интервала важнее справочника.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.active = QLabel()
        self.active.setWordWrap(True)
        try:
            names = [item["name"] for item in source_index(current)["documents"]] if current else []
            self.active.setText("Сейчас: " + (", ".join(names) if names else "основной источник не выбран"))
        except (OSError, ValueError) as exc:
            self.active.setText(str(exc))
        layout.addWidget(self.active)
        self.files = QListWidget()
        layout.addWidget(self.files, 1)
        self.choose = QPushButton("Выбрать материалы…")
        self.choose.clicked.connect(self._choose)
        layout.addWidget(self.choose)
        self.status = QLabel("Word, Excel, текстовый PDF, TXT, Markdown или CSV. Можно выбрать несколько файлов. "
                             "Документы копируются в приложение и работают без интернета; веса VLM при этом не меняются.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.disable = QPushButton("Отключить основной источник")
        self.disable.clicked.connect(self._disable)
        row.addWidget(self.disable)
        row.addStretch()
        self.cancel = QPushButton("Закрыть")
        self.cancel.clicked.connect(self.reject)
        row.addWidget(self.cancel)
        self.apply = QPushButton("Подключить")
        self.apply.setObjectName("primarySourceApply")
        self.apply.clicked.connect(self._import)
        row.addWidget(self.apply)
        layout.addLayout(row)

    def _choose(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Проверенные материалы специалиста", "",
                    "Материалы (*.docx *.xlsx *.xlsm *.pdf *.txt *.md *.csv *.tsv)")
        if files:
            self.files.clear()
            self.files.addItems(files)

    def _disable(self):
        self.folder = ""
        self.accept()

    def _import(self):
        files = [self.files.item(i).text() for i in range(self.files.count())]
        if not files or self.worker is not None:
            self.status.setText("Сначала выберите материалы.")
            return
        folder = user_data_root() / "primary_reference" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        self.import_result = None
        self.worker = WordPhotoJob(import_primary_sources, files, folder, parent=self)
        self.worker.succeeded.connect(self._imported)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self._finished)
        self.worker.finished.connect(self.worker.deleteLater)
        for button in (self.choose, self.apply, self.disable, self.cancel):
            button.setEnabled(False)
        self.status.setText("Читаю документы и создаю локальный справочник…")
        self.worker.start()

    def _imported(self, result):
        self.import_result = result

    def _finished(self):
        self.worker = None
        for button in (self.choose, self.apply, self.disable, self.cancel):
            button.setEnabled(True)
        if self.import_result:
            self.folder = self.import_result["folder"]
            self.accept()

    def reject(self):
        if self.worker is None:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)
