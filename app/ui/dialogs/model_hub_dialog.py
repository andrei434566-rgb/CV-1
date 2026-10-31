from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout
from app.infrastructure.model_hub import read_json
from app.runtime_paths import user_data_root


class ModelHubDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Общие модели и проверенные данные")
        self.resize(780, 560)
        self.setStyleSheet("QWidget {background:#f6faff;color:#153e77;font:13px 'Segoe UI';} QLineEdit,QTextEdit {background:white;border:1px solid #cfdfef;border-radius:7px;padding:8px;} QPushButton {background:#edf5ff;color:#0869c4;border:1px solid #c8e0f8;border-radius:7px;padding:8px 12px;}")
        layout = QVBoxLayout(self)
        note = QLabel("Опубликованные модели копируются в локальный кеш. При недоступности сервера работа продолжается с последней установленной версией. Подтверждённые интервалы передаются в очередь центральной проверки.")
        note.setWordWrap(True); layout.addWidget(note)
        row = QHBoxLayout()
        self.path = QLineEdit(str(main._settings.value("network/hub", "") or ""))
        self.path.setPlaceholderText(r"\\CV1-SERVER\CV1")
        row.addWidget(self.path, 1)
        choose = QPushButton("Папка…"); choose.clicked.connect(self._choose); row.addWidget(choose)
        layout.addLayout(row)
        self.auto_models = QCheckBox("Проверять опубликованные модели при запуске")
        self.auto_models.setChecked(str(main._settings.value("network/auto_models", "true")).lower() != "false")
        layout.addWidget(self.auto_models)
        self.auto_data = QCheckBox("Отправлять подтверждённые интервалы после сохранения проекта")
        self.auto_data.setChecked(str(main._settings.value("network/auto_data", "true")).lower() != "false")
        layout.addWidget(self.auto_data)
        self.status = QTextEdit(); self.status.setReadOnly(True); layout.addWidget(self.status, 1)
        row = QHBoxLayout()
        for label, action in (("Проверить модели", "sync"), ("Отправить проверенные данные", "submit"), ("Предыдущая версия", "rollback")):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, op=action: self._run(op))
            row.addWidget(button)
        layout.addLayout(row)
        stop = QPushButton("Остановить текущую синхронизацию")
        stop.clicked.connect(main._hub_client.stop)
        layout.addWidget(stop)
        row = QHBoxLayout(); row.addStretch()
        save = QPushButton("Сохранить"); save.clicked.connect(self._save); row.addWidget(save)
        close = QPushButton("Закрыть"); close.clicked.connect(self.reject); row.addWidget(close)
        layout.addLayout(row)
        main._hub_client.status_changed.connect(self.status.append)
        main._hub_client.finished.connect(self._result)
        self._refresh()

    def _choose(self):
        folder = QFileDialog.getExistingDirectory(self, "Общая папка CV-1", self.path.text())
        if folder: self.path.setText(folder)

    def _persist(self):
        self.main._settings.setValue("network/hub", self.path.text().strip())
        self.main._settings.setValue("network/auto_models", self.auto_models.isChecked())
        self.main._settings.setValue("network/auto_data", self.auto_data.isChecked())
        self.main._settings.sync()

    def _save(self):
        self._persist(); self.accept()

    def _run(self, operation):
        self._persist()
        self.main._network_action(operation)

    def _result(self, operation, result):
        self.status.append(("Готово: " + operation) if result.get("ok") else ("Не выполнено: " + result.get("error", "")))
        self._refresh()

    def _refresh(self):
        active = user_data_root() / "network_cache/active.json"
        if not active.exists():
            self.status.append("Сетевые версии ещё не установлены. Используются ваши локальные модели.")
            return
        try:
            data = read_json(active)
            self.status.append("Локальный набор: " + data["release_id"])
            for model in data["models"]:
                scope = model.get("approval", {}).get("scope", "")
                self.status.append(f"{model['role']}: {model['version']} · {scope}")
        except (OSError, ValueError, KeyError) as exc:
            self.status.append(str(exc))
