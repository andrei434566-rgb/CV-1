"""Local model paths and knowledge folders; nothing is uploaded."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QVBoxLayout, QWidget

from app.infrastructure.local_vlm_modes import normalize_model_request


class LocalVlmSettingsDialog(QDialog):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Локальный ИИ и папки знаний")
        self.resize(700, 560)
        self._path_widgets: dict[QLineEdit, QWidget] = {}
        layout = QVBoxLayout(self)
        note = QLabel("Модель работает на этом компьютере. Интернет и сервер не используются. Данные из выбранных папок читаются только для поиска.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.model_mode = QComboBox()
        self.model_mode.setObjectName("localModelMode")
        self.model_mode.addItem("Базовая модель", "base")
        self.model_mode.addItem("Модель с дообучением (LoRA)", "lora")
        initial = str(config.get("model_mode") or "").casefold()
        if initial not in {"base", "lora"}:
            initial = "lora" if config.get("adapter") or config.get("theory_adapter") else "base"
        self.model_mode.setCurrentIndex(self.model_mode.findData(initial))
        form.addRow("Режим ИИ:", self.model_mode)
        self.python = self._path_row(form, "Локальный Python:", str(config.get("python") or ""), file=True)
        self.base = self._path_row(form, "Базовая Qwen3-VL:", str(config.get("base") or ""))
        self.adapter = self._path_row(form, "Папка дообучения:", str(config.get("adapter") or ""))
        self.separate_text = QCheckBox("Для общения использовать отдельный адаптер")
        self.separate_text.setChecked(bool(config.get("theory_adapter")))
        form.addRow("", self.separate_text)
        self.theory_adapter = self._path_row(form, "Адаптер для общения:", str(config.get("theory_adapter") or ""))
        self.model_mode.currentIndexChanged.connect(self._update_mode_fields)
        self.separate_text.toggled.connect(self._update_mode_fields)
        self._update_mode_fields()
        layout.addLayout(form)
        resources = QLabel(
            "Базовая модель подходит для проверки работы приложения. В режиме с дообучением "
            "она загружается вместе с выбранным адаптером: веса остаются раздельными. "
            "Используйте ту же базовую Qwen3-VL и версию, на которой обучался адаптер. "
            "По умолчанию один адаптер используется для фото и общения. "
            "Приложение читает готовые локальные файлы и ничего не скачивает."
        )
        resources.setObjectName("vlmResources")
        resources.setWordWrap(True)
        layout.addWidget(resources)
        layout.setSpacing(12)
        self.setStyleSheet(self.styleSheet() + """
            QDialog { background: #f7fbff; color: #163d71; }
            QLabel#vlmResources { background: #edf5ff; border: 1px solid #d1e6fb;
                border-radius: 10px; padding: 12px; color: #163d71; }
            QLabel#vlmResources a { color: #087ce8; }
            QLineEdit, QListWidget, QComboBox { background: white; border: 1px solid #c7ddf2;
                border-radius: 8px; padding: 7px; color: #163d71; }
            QPushButton { background: #e6f3ff; border: 1px solid #b9dafa;
                border-radius: 8px; padding: 7px 12px; color: #0b5ca8; }
            QPushButton:hover { background: #d3eaff; }
        """)
        layout.addWidget(QLabel("Папки для поиска ключевых слов:"))
        self.folders = QListWidget()
        for folder in config.get("folders") or []:
            self.folders.addItem(str(folder))
        layout.addWidget(self.folders, 1)
        buttons = QHBoxLayout()
        add = QPushButton("Добавить папку…")
        add.clicked.connect(self._add_folder)
        buttons.addWidget(add)
        remove = QPushButton("Убрать выбранную")
        remove.clicked.connect(lambda: self.folders.takeItem(self.folders.currentRow()))
        buttons.addWidget(remove)
        buttons.addStretch()
        layout.addLayout(buttons)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)

    def _path_row(self, form: QFormLayout, label: str, value: str, *, file: bool = False) -> QLineEdit:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(value)
        row.addWidget(edit, 1)
        browse = QPushButton("Выбрать…")
        browse.clicked.connect(lambda: self._browse(edit, file))
        row.addWidget(browse)
        form.addRow(label, widget)
        self._path_widgets[edit] = widget
        return edit

    def _update_mode_fields(self, *_args) -> None:
        use_lora = self.model_mode.currentData() == "lora"
        self._path_widgets[self.adapter].setEnabled(use_lora)
        self.separate_text.setEnabled(use_lora)
        self._path_widgets[self.theory_adapter].setEnabled(use_lora and self.separate_text.isChecked())

    def accept(self) -> None:
        try:
            normalize_model_request(self.values())
        except ValueError as exc:
            QMessageBox.warning(self, "Настройки локального ИИ", str(exc))
            return
        super().accept()

    def _browse(self, edit: QLineEdit, file: bool) -> None:
        if file:
            selected, _ = QFileDialog.getOpenFileName(self, "Выбрать локальный Python", edit.text(), "Python (python.exe);;Все файлы (*)")
        else:
            selected = QFileDialog.getExistingDirectory(self, "Выбрать локальную папку", edit.text())
        if selected:
            edit.setText(selected)

    def _add_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Папка с локальными материалами")
        if selected and all(self.folders.item(i).text() != selected for i in range(self.folders.count())):
            self.folders.addItem(selected)

    def values(self) -> dict:
        return {
            "model_mode": str(self.model_mode.currentData()),
            "python": self.python.text().strip(),
            "base": self.base.text().strip(),
            "adapter": self.adapter.text().strip(),
            "theory_adapter": self.theory_adapter.text().strip() if self.separate_text.isChecked() else "",
            "folders": [self.folders.item(i).text() for i in range(self.folders.count())],
        }
