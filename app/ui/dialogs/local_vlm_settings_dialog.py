"""Local model paths and knowledge folders; nothing is uploaded."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QVBoxLayout


class LocalVlmSettingsDialog(QDialog):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Локальная VLM и папки знаний")
        self.resize(680, 480)
        layout = QVBoxLayout(self)
        note = QLabel("Модель работает на этом компьютере. Интернет и сервер не используются. Данные из выбранных папок читаются только для поиска.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.python = self._path_row(form, "Python VLM:", str(config.get("python") or ""), file=True)
        self.base = self._path_row(form, "Qwen3-VL-2B:", str(config.get("base") or ""))
        self.adapter = self._path_row(form, "Адаптер LoRA:", str(config.get("adapter") or ""))
        self.theory_adapter = self._path_row(form, "Текстовый LoRA для чата:", str(config.get("theory_adapter") or ""))
        layout.addLayout(form)
        resources = QLabel(
            '<b>Ссылки для установки и тестирования VLM</b><br>'
            'Базовая модель: <a href="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct">'
            'Qwen3-VL-2B-Instruct на Hugging Face</a><br>'
            'Инструкция и примеры: <a href="https://github.com/QwenLM/Qwen3-VL">'
            'официальный Qwen3-VL</a><br>'
            'Свой LoRA-адаптер: укажите локальную папку с обученными весами в поле выше. '
            'Ссылка на базовую модель не скачивает и не подключает адаптер автоматически.'
        )
        resources.setObjectName("vlmResources")
        resources.setOpenExternalLinks(True)
        resources.setWordWrap(True)
        layout.addWidget(resources)
        layout.setSpacing(12)
        self.setStyleSheet(self.styleSheet() + """
            QDialog { background: #f7fbff; color: #163d71; }
            QLabel#vlmResources { background: #edf5ff; border: 1px solid #d1e6fb;
                border-radius: 10px; padding: 12px; color: #163d71; }
            QLabel#vlmResources a { color: #087ce8; }
            QLineEdit, QListWidget { background: white; border: 1px solid #c7ddf2;
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
        row = QHBoxLayout()
        edit = QLineEdit(value)
        row.addWidget(edit, 1)
        browse = QPushButton("Выбрать…")
        browse.clicked.connect(lambda: self._browse(edit, file))
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

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
            "python": self.python.text().strip(),
            "base": self.base.text().strip(),
            "adapter": self.adapter.text().strip(),
            "theory_adapter": self.theory_adapter.text().strip(),
            "folders": [self.folders.item(i).text() for i in range(self.folders.count())],
        }
