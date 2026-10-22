"""Review original open-source photos and write the specialist's Russian answer."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from app.infrastructure.open_dataset_review import export_reviewed_pairs, read_candidates, save_review


class OpenDatasetReviewDialog(QDialog):
    pairs_ready = Signal(dict)

    def __init__(self, preparation: dict, parent=None):
        super().__init__(parent)
        self.preparation = preparation
        self.rows = read_candidates(preparation)
        self.index = -1
        self._zoom = 1.0
        self._original_photo = QPixmap()
        self.setWindowTitle("Проверка фото и описаний")
        self.resize(1120, 800)
        layout = QVBoxLayout(self)
        introduction = QLabel(
            "Сравните фото с исходной записью и напишите свой ответ по-русски. "
            "Подтверждайте только признаки, которые видны в этом интервале. "
            "Исходный код класса не заменяет проверенное описание."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.selector = QComboBox()
        for number, row in enumerate(self.rows, 1):
            metadata = row["metadata"]
            self.selector.addItem(f"{number}. {metadata.get('well_id', 'Скважина не указана')} — {row.get('source_label', '')}")
        layout.addWidget(self.selector)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        photo_panel = QWidget()
        photo_layout = QVBoxLayout(photo_panel)
        photo_controls = QHBoxLayout()
        for title, factor in (("Уменьшить", 0.8), ("Увеличить", 1.25)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, f=factor: self._change_zoom(f))
            photo_controls.addWidget(button)
        photo_layout.addLayout(photo_controls)
        self.photo = QLabel()
        self.photo.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        photo_scroll = QScrollArea()
        photo_scroll.setWidget(self.photo)
        photo_scroll.setWidgetResizable(False)
        photo_layout.addWidget(photo_scroll)
        splitter.addWidget(photo_panel)
        answer_panel = QWidget()
        answer_layout = QVBoxLayout(answer_panel)
        answer_layout.addWidget(QLabel("Исходная запись — сохраняется без изменений"))
        self.original = QTextEdit()
        self.original.setReadOnly(True)
        self.original.setMaximumHeight(235)
        answer_layout.addWidget(self.original)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color:#984b00;")
        answer_layout.addWidget(self.warning)
        answer_layout.addWidget(QLabel("Ваш проверенный ответ на русском языке"))
        self.answer = QTextEdit()
        self.answer.setPlaceholderText("Напишите описание по этому фото. Укажите неопределённость, если признак неразличим.")
        answer_layout.addWidget(self.answer)
        self.confirm = QCheckBox("Я проверил фото, границы интервала и каждое утверждение ответа")
        self.confirm.setChecked(False)
        answer_layout.addWidget(self.confirm)
        save_buttons = QHBoxLayout()
        draft = QPushButton("Сохранить черновик")
        draft.clicked.connect(self._save_draft_explicit)
        save_buttons.addWidget(draft)
        self.approve = QPushButton("Подтвердить пару")
        self.approve.clicked.connect(self._approve)
        save_buttons.addWidget(self.approve)
        answer_layout.addLayout(save_buttons)
        splitter.addWidget(answer_panel)
        splitter.setSizes([500, 600])
        layout.addWidget(splitter, 1)
        navigation = QHBoxLayout()
        previous = QPushButton("Назад")
        previous.clicked.connect(lambda: self._navigate(-1))
        navigation.addWidget(previous)
        following = QPushButton("Следующая / пропустить")
        following.clicked.connect(lambda: self._navigate(1))
        navigation.addWidget(following)
        navigation.addStretch()
        export = QPushButton("Передать проверенные пары для обучения")
        export.clicked.connect(self._export)
        navigation.addWidget(export)
        layout.addLayout(navigation)
        self.selector.currentIndexChanged.connect(self._select)
        self.answer.textChanged.connect(self._answer_changed)
        if self.rows:
            self._select(0)
        else:
            self.status.setText("В источнике нет сопоставленных фото. Документы доступны в библиотеке материалов.")
            self.approve.setEnabled(False)
            export.setEnabled(False)

    def _show_error(self, exc):
        QMessageBox.warning(self, "Проверка пары", str(exc))

    def _current(self):
        return self.rows[self.index] if 0 <= self.index < len(self.rows) else None

    def _answer_changed(self):
        self.confirm.setChecked(False)

    def _persist_draft(self):
        row = self._current()
        if row and self.answer.document().isModified():
            row["review"] = save_review(self.preparation, row["record_id"], self.answer.toPlainText(), False)
            self.answer.document().setModified(False)
        return True

    def _save_draft_explicit(self):
        try:
            row = self._current()
            if row:
                row["review"] = save_review(self.preparation, row["record_id"], self.answer.toPlainText(), False)
                self.answer.document().setModified(False)
                self.confirm.setChecked(False)
                self._update_status()
        except (OSError, ValueError) as exc:
            self._show_error(exc)

    def _select(self, index):
        if not 0 <= index < len(self.rows) or index == self.index:
            return
        try:
            self._persist_draft()
        except (OSError, ValueError) as exc:
            self.selector.blockSignals(True)
            self.selector.setCurrentIndex(self.index)
            self.selector.blockSignals(False)
            self._show_error(exc)
            return
        self.index = index
        row = self.rows[index]
        meta, review = row["metadata"], row.get("review", {})
        self.answer.setPlainText(review.get("answer_ru", ""))
        self.answer.document().setModified(False)
        self.confirm.setChecked(False)
        self.original.setPlainText(
            f"Скважина: {meta.get('well_id', 'не указана')}\n"
            f"Интервал фото: {meta.get('depth_top_m', '?')}–{meta.get('depth_bottom_m', '?')} м\n"
            f"Шкала глубины: {meta.get('depth_reference', 'не указана')}\n"
            f"Исходный класс: {row.get('source_label', '')}\n"
            f"Исходное описание: {row.get('original_description', '')}\n"
            f"Источник: {meta.get('source_path', '')}\n"
            f"Запись: {meta.get('source_member', '')} {meta.get('source_row', '')}"
        )
        needs_crop = bool(meta.get("needs_interval_crop"))
        if needs_crop:
            self.warning.setText("Это фото охватывает более длинный интервал. Подтвердить пару можно после подготовки отдельной вырезки слоя.")
        elif review.get("source_changed"):
            self.warning.setText("Исходная запись изменилась. Старое подтверждение отменено; проверьте пару повторно.")
        else:
            self.warning.setText("Подтверждено специалистом" if review.get("approved") else "Требуется проверка фото и русский ответ специалиста")
        self._original_photo = QPixmap(row["image_path"])
        self._zoom = min(1.0, 500 / max(1, self._original_photo.width()))
        self._render_photo()
        self.approve.setEnabled(not needs_crop and not self._original_photo.isNull())
        self._update_status()

    def _render_photo(self):
        if self._original_photo.isNull():
            self.photo.setText("Фото не найдено или не читается")
            self.photo.adjustSize()
            return
        size = self._original_photo.size() * self._zoom
        photo = self._original_photo.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.photo.setPixmap(photo)
        self.photo.resize(photo.size())

    def _change_zoom(self, factor):
        self._zoom = min(4.0, max(0.1, self._zoom * factor))
        self._render_photo()

    def _navigate(self, direction):
        self.selector.setCurrentIndex(min(len(self.rows) - 1, max(0, self.index + direction)))

    def _update_status(self):
        confirmed = sum(bool(row.get("review", {}).get("approved")) for row in self.rows)
        wells = {str(row["metadata"].get("well_id", "")).strip().casefold()
                 for row in self.rows if row.get("review", {}).get("approved")}
        self.status.setText(f"Пара {self.index + 1} из {len(self.rows)}. Подтверждено: {confirmed}; скважин: {len(wells - {''})}. Для экспорта нужны минимум две скважины.")

    def _approve(self):
        if not self.confirm.isChecked():
            QMessageBox.information(self, "Подтверждение", "Поставьте отметку, что вы проверили фото, границы и ответ")
            return
        row = self._current()
        if not row:
            return
        try:
            row["review"] = save_review(self.preparation, row["record_id"], self.answer.toPlainText(), True)
            self.answer.document().setModified(False)
            self.warning.setText("Подтверждено специалистом")
            self._update_status()
            self._navigate(1)
        except (OSError, ValueError) as exc:
            self._show_error(exc)

    def _export(self):
        try:
            self._persist_draft()
            result = export_reviewed_pairs(self.preparation)
            self.pairs_ready.emit(result)
            QMessageBox.information(self, "Пары подготовлены",
                                    f"Обучение: {result['counts']['train']} пар. Независимый контроль: {result['counts']['holdout']} пар.\n"
                                    "Материалы переданы в окно обучения; обучение ещё не запущено.")
        except (OSError, ValueError) as exc:
            self._show_error(exc)

    def reject(self):
        try:
            self._persist_draft()
        except (OSError, ValueError) as exc:
            self._show_error(exc)
            return
        super().reject()
