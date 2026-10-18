"""One local entry point for the three independent learning workflows."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)

from app.infrastructure.expert_corrections import correction_counts, export_correction_manifest
from app.infrastructure.vlm_training_data import export_reviewed_pairs, export_theory_qa, reviewed_pair_counts
from app.runtime_paths import bundled_root, user_data_root


class TrainingCenterDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main = main_window
        self.settings: QSettings = main_window._settings
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self._track = ""
        self._output_dir: Path | None = None
        self.resize(780, 690)
        self.setWindowTitle("Обучение · фото + Excel, теория, сегментация")
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        intro = QLabel(
            "Три независимых потока. Фото + Excel обучают Qwen описывать видимые признаки; "
            "теория пополняет локальный поиск или отдельный текстовый LoRA; "
            "проверенные контуры обучают YOLO сегментации."
        )
        intro.setWordWrap(True)
        content_layout.addWidget(intro)

        catalog_box = QGroupBox("Основной справочник фаций для YOLO")
        catalog_layout = QVBoxLayout(catalog_box)
        self.catalog_status = QLabel()
        self.catalog_status.setWordWrap(True)
        catalog_layout.addWidget(self.catalog_status)
        choose_catalog = QPushButton("Загрузить Excel со всеми фациями…")
        choose_catalog.clicked.connect(self._import_catalog)
        catalog_layout.addWidget(choose_catalog)
        content_layout.addWidget(catalog_box)

        pairs = QGroupBox("1. Проверенные пары фото + Excel → Qwen3-VL")
        p = QVBoxLayout(pairs)
        self.pair_status = QLabel()
        p.addWidget(self.pair_status)
        p.addWidget(QLabel("Импорт сопоставляет глубины; затем проверьте контуры и описание каждого слоя в карточке фации."))
        row = QHBoxLayout()
        self.import_pairs = QPushButton("Excel + папка фото…")
        self.import_pairs.clicked.connect(self._import_pairs)
        row.addWidget(self.import_pairs)
        choose_photos = QPushButton("Excel + выбранные фото…")
        choose_photos.clicked.connect(self._import_selected_photos)
        row.addWidget(choose_photos)
        self.export_pairs = QPushButton("Собрать проверенные пары")
        self.export_pairs.clicked.connect(self._export_pairs)
        row.addWidget(self.export_pairs)
        p.addLayout(row)
        self.photo_feedback_button = QPushButton("Собрать поправки по фото из чата")
        self.photo_feedback_button.clicked.connect(lambda: self._export_feedback("pairs"))
        p.addWidget(self.photo_feedback_button)
        self.pair_manifest = QLabel("Манифест: —")
        self.pair_manifest.setWordWrap(True)
        p.addWidget(self.pair_manifest)
        self.continue_adapter = QCheckBox("Продолжить с активного LoRA описаний")
        self.continue_adapter.setChecked(True)
        p.addWidget(self.continue_adapter)
        self.train_pairs = QPushButton("Дообучить Qwen на парах")
        self.train_pairs.clicked.connect(lambda: self._train("pairs"))
        p.addWidget(self.train_pairs)
        content_layout.addWidget(pairs)

        theory = QGroupBox("2. Теория → локальный поиск или отдельный текстовый LoRA")
        t = QVBoxLayout(theory)
        explanation = QLabel(
            "PDF/DOCX/TXT/Excel можно добавить в папки знаний для поиска по источникам. "
            "Для дообучения используйте проверенную .xlsx/.csv таблицу с колонками «Вопрос» и «Ответ». "
            "Свободный текст не превращается автоматически в выдуманные ответы."
        )
        explanation.setWordWrap(True)
        t.addWidget(explanation)
        row = QHBoxLayout()
        add_folder = QPushButton("Добавить папку теории для поиска…")
        add_folder.clicked.connect(self._add_theory_folder)
        row.addWidget(add_folder)
        import_qa = QPushButton("Загрузить вопросы и ответы…")
        import_qa.clicked.connect(self._import_theory)
        row.addWidget(import_qa)
        t.addLayout(row)
        self.text_feedback_button = QPushButton("Собрать общие поправки из чата")
        self.text_feedback_button.clicked.connect(lambda: self._export_feedback("theory"))
        t.addWidget(self.text_feedback_button)
        self.feedback_status = QLabel()
        t.addWidget(self.feedback_status)
        self.theory_manifest = QLabel("Текстовый манифест: —")
        self.theory_manifest.setWordWrap(True)
        t.addWidget(self.theory_manifest)
        self.continue_theory_adapter = QCheckBox("Продолжить с активного текстового LoRA")
        theory_adapter = Path(self.main._local_vlm_config()["theory_adapter"])
        self.continue_theory_adapter.setChecked((theory_adapter / "adapter_config.json").is_file())
        t.addWidget(self.continue_theory_adapter)
        self.train_theory = QPushButton("Дообучить отдельный текстовый LoRA")
        self.train_theory.clicked.connect(lambda: self._train("theory"))
        t.addWidget(self.train_theory)
        content_layout.addWidget(theory)

        yolo = QGroupBox("3. Предобученная .pt → YOLO сегментации")
        y = QVBoxLayout(yolo)
        self.yolo_status = QLabel()
        self.yolo_status.setWordWrap(True)
        y.addWidget(self.yolo_status)
        row = QHBoxLayout()
        choose = QPushButton("Выбрать исходную .pt…")
        choose.clicked.connect(self._choose_yolo)
        row.addWidget(choose)
        train = QPushButton("Обучить на проверенных контурах")
        train.clicked.connect(self._train_yolo)
        row.addWidget(train)
        y.addLayout(row)
        content_layout.addWidget(yolo)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Ход подготовки и обучения будет показан здесь.")
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)
        self.refresh()

    def refresh(self) -> None:
        catalog = self.main._primary_facies_catalog
        self.catalog_status.setText(
            f"Основной: {catalog['source_name']} · {len(catalog['classes'])} классов. Снимок сохраняется с проектом."
            if catalog else "Сейчас используется встроенный справочник. Свой Excel можно добавить позже на этом ПК."
        )
        reviewed, usable = reviewed_pair_counts(self.main._records)
        self.pair_status.setText(f"Текущий проект: проверено слоёв {reviewed}; из них с описанием Excel {usable}.")
        corrections = correction_counts()
        self.feedback_status.setText(
            f"Поправки специалиста: текстовых {corrections['text']} "
            f"(общих для обучения {corrections['text_global']}), по фото {corrections['photo']}."
        )
        model = self.main._resolve_model_path()
        self.yolo_status.setText(
            f"Исходная модель: {model if model else 'не выбрана'}\n"
            f"Для YOLO нужны проверенные контуры на фото, одних строк Excel недостаточно."
        )
        for track, label in (("pairs", self.pair_manifest), ("theory", self.theory_manifest)):
            value = str(self.settings.value(f"training/{track}_manifest", "") or "")
            label.setText(("Манифест: " if track == "pairs" else "Текстовый манифест: ") + (value if Path(value).is_file() else "—"))

    def _import_pairs(self) -> None:
        self.main.import_excel_photo_batch()
        self.refresh()

    def _import_selected_photos(self) -> None:
        self.main.import_excel_photo_files()
        self.refresh()

    def _import_catalog(self) -> None:
        self.main.import_primary_facies_catalog()
        self.refresh()

    def _export_pairs(self) -> None:
        root = user_data_root() / "vlm_training" / "photo_excel"
        destination = root / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        try:
            result = export_reviewed_pairs(self.main._records, destination)
        except Exception as exc:
            QMessageBox.warning(self, "Пары не собраны", str(exc))
            return
        self.settings.setValue("training/pairs_manifest", str(result["manifest"]))
        self.settings.remove("training/pairs_additional_manifest")
        self.settings.sync()
        self.log.append(f"Сохранено проверенных пар: {result['count']}; пропущено: {result['skipped']}. {result['manifest']}")
        self.refresh()

    def _add_theory_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка с теорией")
        if not folder:
            return
        config = self.main._local_vlm_config()
        folders = list(config["folders"])
        if folder not in folders:
            folders.append(folder)
        self.settings.setValue("vlm/knowledge_folders_json", json.dumps(folders, ensure_ascii=False))
        self.settings.sync()
        self.log.append(f"Добавлена папка для локального поиска: {folder}")

    def _import_theory(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Теория: таблица вопрос–ответ", "", "Excel/CSV (*.xlsx *.csv)")
        if not filename:
            return
        destination = user_data_root() / "vlm_training" / "theory" / f"theory_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.jsonl"
        try:
            result = export_theory_qa(Path(filename), destination)
        except Exception as exc:
            QMessageBox.warning(self, "Теория не загружена", str(exc))
            return
        self.settings.setValue("training/theory_manifest", str(result["manifest"]))
        self.settings.remove("training/theory_additional_manifest")
        self.settings.sync()
        self.log.append(f"Сохранено проверяемых текстовых пар: {result['count']}. {destination}")
        self.refresh()

    def _export_feedback(self, track: str) -> None:
        kind = "photo" if track == "pairs" else "text"
        destination = (
            user_data_root() / "vlm_training" / "expert_feedback" / "exports"
            / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.jsonl"
        )
        try:
            result = export_correction_manifest(kind, destination)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Поправки не собраны", str(exc))
            return
        previous = Path(str(self.settings.value(f"training/{track}_manifest", "") or ""))
        if previous.is_file() and previous.parent != destination.parent:
            self.settings.setValue(f"training/{track}_additional_manifest", str(previous))
        self.settings.setValue(f"training/{track}_manifest", str(result["manifest"]))
        self.settings.sync()
        self.log.append(f"Подготовлено поправок для {track}: {result['count']}. {destination}")
        self.refresh()

    def _choose_yolo(self) -> None:
        self.main.select_detection_model()
        self.refresh()

    def _train_yolo(self) -> None:
        self.main.start_fine_tune()
        self.refresh()

    def _train(self, track: str) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Обучение", "Дождитесь завершения текущего обучения Qwen.")
            return
        manifest = Path(str(self.settings.value(f"training/{track}_manifest", "") or ""))
        config = self.main._local_vlm_config()
        python = Path(config["python"])
        base = Path(config["base"])
        script = bundled_root() / "app" / "infrastructure" / "ml" / "qwen_train.py"
        if not manifest.is_file() or not python.is_file() or not (base / "config.json").is_file() or not script.is_file():
            QMessageBox.warning(self, "Обучение Qwen", "Проверьте манифест, локальный Python и папку базовой Qwen в настройках ИИ.")
            return
        self._output_dir = user_data_root() / "vlm_models" / track / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        args = ["-u", str(script), "--model", str(base), "--manifest", str(manifest), "--output-dir", str(self._output_dir)]
        additional = Path(str(self.settings.value(f"training/{track}_additional_manifest", "") or ""))
        if additional.is_file() and additional != manifest:
            args += ["--additional-manifest", str(additional)]
        if track == "pairs" and self.continue_adapter.isChecked():
            adapter = Path(config["adapter"])
            if not (adapter / "adapter_config.json").is_file():
                QMessageBox.warning(self, "Начальный LoRA", "Текущий адаптер не найден. Укажите его в настройках ИИ или снимите галочку продолжения.")
                return
            args += ["--initial-adapter", str(adapter)]
        if track == "theory" and self.continue_theory_adapter.isChecked():
            adapter = Path(config["theory_adapter"])
            if not (adapter / "adapter_config.json").is_file():
                QMessageBox.warning(self, "Начальный LoRA", "Текущий текстовый адаптер не найден. Укажите его в настройках ИИ или снимите галочку продолжения.")
                return
            args += ["--initial-adapter", str(adapter)]
        if self.main._vlm_pending is not None:
            QMessageBox.information(self, "Обучение Qwen", "Дождитесь текущего ответа ассистента и повторите запуск обучения.")
            return
        self.main._vlm_client.stop()
        self._track = track
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("HF_HUB_OFFLINE", "1")
        environment.insert("TRANSFORMERS_OFFLINE", "1")
        self.process.setProcessEnvironment(environment)
        self.process.start(str(python), args)
        if not self.process.waitForStarted(5000):
            QMessageBox.critical(self, "Обучение Qwen", self.process.errorString())
            return
        self.train_pairs.setEnabled(False)
        self.train_theory.setEnabled(False)
        self.close_button.setEnabled(False)
        self.log.append(f"Запущено локальное обучение {track}: {manifest}")

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data.strip():
            self.log.append(data.strip()[-2500:])

    def _finished(self, code: int, _status) -> None:
        self._read_output()
        self.train_pairs.setEnabled(True)
        self.train_theory.setEnabled(True)
        self.close_button.setEnabled(True)
        output = self._output_dir
        if code == 0 and output and (output / "adapter_config.json").is_file():
            self.settings.setValue(f"training/{self._track}_candidate_adapter", str(output))
            self.settings.sync()
            self.log.append(f"Кандидат LoRA готов: {output}. Проверьте на независимых данных перед включением.")
            QMessageBox.information(self, "Кандидат LoRA готов", f"{output}\n\nПроверьте ответы на отдельных скважинах. Затем выберите адаптер в настройках ИИ.")
        else:
            self.log.append(f"Обучение завершилось с кодом {code}. Проверьте журнал и локальную CUDA-память.")

    def closeEvent(self, event) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Обучение идёт", "Дождитесь окончания обучения Qwen, прежде чем закрывать окно.")
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Обучение идёт", "Дождитесь окончания обучения Qwen, прежде чем закрывать окно.")
            return
        super().reject()
