from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QTextEdit, QVBoxLayout
from app.infrastructure.model_hub import read_json
from app.infrastructure.network_settings import normalize_hub_path
from app.runtime_paths import user_data_root
from app.ui.dialog_theme import apply_light_card


class ModelHubDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Серверная папка · общие модели и данные")
        self.resize(820, 650)
        apply_light_card(self)
        layout = QVBoxLayout(self)
        note = QLabel("Опубликованные модели копируются в локальный кеш. При недоступности сервера работа продолжается с последней установленной версией. Подтверждённые интервалы передаются в очередь центральной проверки.")
        note.setWordWrap(True); layout.addWidget(note)
        later = QLabel("Папку можно указать позже на другом ПК. Введите адрес вручную или выберите в Проводнике. "
                       "Сохранение настройки не требует подключения к серверу.")
        later.setWordWrap(True); layout.addWidget(later)
        layout.addWidget(QLabel("Папка обмена CV-1 на сервере:"))
        row = QHBoxLayout()
        self.path = QLineEdit(str(main._settings.value("network/hub", "") or ""))
        self.path.setPlaceholderText(r"\\SERVER\Общая папка\CV1 или Z:\CV1")
        self.path.setObjectName('serverFolderPath')
        row.addWidget(self.path, 1)
        choose = QPushButton("Выбрать папку…"); choose.clicked.connect(self._choose); row.addWidget(choose)
        layout.addLayout(row)
        self.enabled = QCheckBox('Использовать серверную папку на этом ПК')
        self.enabled.setChecked(str(main._settings.value('network/enabled', 'true')).lower() != 'false')
        layout.addWidget(self.enabled)
        self.auto_models = QCheckBox("Проверять опубликованные модели при запуске")
        self.auto_models.setChecked(str(main._settings.value("network/auto_models", "true")).lower() != "false")
        layout.addWidget(self.auto_models)
        self.auto_data = QCheckBox("Отправлять подтверждённые интервалы после сохранения проекта")
        self.auto_data.setChecked(str(main._settings.value("network/auto_data", "true")).lower() != "false")
        layout.addWidget(self.auto_data)
        row = QHBoxLayout()
        for label, action in (("Проверить доступ", "probe"), ("Подготовить папку для CV-1", "initialize")):
            button = QPushButton(label); button.clicked.connect(lambda _checked=False, op=action: self._run(op))
            row.addWidget(button)
        layout.addLayout(row)
        setup_note = QLabel('Подготовка создаёт папки обмена и служебный файл CV-1. '
                            'Модели публикуются отдельно после проверки. Для подготовки нужны права записи в выбранную папку.')
        setup_note.setWordWrap(True); layout.addWidget(setup_note)
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
        save = QPushButton("Сохранить настройки"); save.setObjectName('saveServerFolder'); save.clicked.connect(self._save); row.addWidget(save)
        close = QPushButton("Закрыть"); close.clicked.connect(self.reject); row.addWidget(close)
        layout.addLayout(row)
        main._hub_client.status_changed.connect(self.status.append)
        main._hub_client.finished.connect(self._result)
        self._refresh()

    def _choose(self):
        # Do not open a potentially disconnected UNC path just to show a picker.
        folder = QFileDialog.getExistingDirectory(self, "Папка обмена CV-1 на сервере")
        if folder: self.path.setText(folder)

    def _persist(self):
        try:
            value = normalize_hub_path(self.path.text())
        except ValueError as exc:
            QMessageBox.warning(self, 'Проверьте адрес папки', str(exc)); return False
        previous = str(self.main._settings.value('network/hub', '') or '')
        if self.main._hub_client.busy and previous.casefold() != value.casefold():
            self.status.append('Сначала остановите текущую синхронизацию, затем смените серверную папку.')
            return False
        self.main._settings.setValue("network/hub", value)
        self.main._settings.setValue('network/enabled', self.enabled.isChecked())
        if previous.casefold() != value.casefold():
            self.main._settings.remove('network/last_submitted_signature')
        self.main._settings.setValue("network/auto_models", self.auto_models.isChecked())
        self.main._settings.setValue("network/auto_data", self.auto_data.isChecked())
        self.main._settings.sync()
        self.path.setText(value)
        return True

    def _save(self):
        if self._persist():
            self.main.statusBar().showMessage('Настройки серверной папки сохранены. Подключение можно проверить позже.')
            self.accept()

    def _run(self, operation):
        if not self._persist(): return
        if not self.path.text() and operation != 'rollback':
            self.status.append('Сначала укажите папку обмена. Сохранить пустую настройку и настроить позже тоже можно.'); return
        if not self.enabled.isChecked() and operation in {'sync', 'submit'}:
            self.status.append('Включите использование серверной папки, чтобы начать обмен.'); return
        self.main._network_action(operation)

    def _result(self, operation, result):
        self.status.append((result.get('message') or "Операция завершена: " + operation) if result.get("ok") else ("Не выполнено: " + result.get("error", "")))
        self._refresh()

    def showEvent(self, event):
        self.path.setText(str(self.main._settings.value('network/hub', '') or ''))
        self.enabled.setChecked(str(self.main._settings.value('network/enabled', 'true')).lower() != 'false')
        self.auto_models.setChecked(str(self.main._settings.value('network/auto_models', 'true')).lower() != 'false')
        self.auto_data.setChecked(str(self.main._settings.value('network/auto_data', 'true')).lower() != 'false')
        super().showEvent(event)

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
