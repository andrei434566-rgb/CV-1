"""Right-hand offline assistant pane for the Kern Analyzer workspace."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

from PySide6.QtCore import QSize, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


class MessageComposer(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class PhotoPreviewLabel(QLabel):
    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


class AssistantPanel(QFrame):
    send_requested = Signal(str)
    correction_requested = Signal(object)
    files_requested = Signal()
    files_dropped = Signal(object)
    found_photo_requested = Signal(str)
    photo_clear_requested = Signal()
    photo_analysis_requested = Signal()
    expand_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assistantPanel")
        self.setMinimumWidth(290)
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(12)

        title_row = QHBoxLayout()
        title = QLabel("ИИ-ассистент")
        title.setObjectName("assistantTitle")
        title_row.addWidget(title, 1)
        self.scroll_up_button = QPushButton("↑")
        self.scroll_up_button.setToolTip("Прокрутить переписку вверх")
        self.scroll_up_button.clicked.connect(lambda: self._scroll_messages(-1))
        self.scroll_up_button.hide()
        self.scroll_down_button = QPushButton("↓")
        self.scroll_down_button.setToolTip("Прокрутить переписку вниз")
        self.scroll_down_button.clicked.connect(lambda: self._scroll_messages(1))
        self.scroll_down_button.hide()
        self.expand_button = QPushButton("Шире")
        self.expand_button.setToolTip("Расширить или сузить чат; ширину также можно менять левой границей")
        self.expand_button.clicked.connect(self.expand_requested)
        title_row.addWidget(self.expand_button)
        layout.addLayout(title_row)
        self.status = QLabel("●  Готов к работе")
        self.status.setObjectName("assistantStatus")
        layout.addWidget(self.status)
        self.context = QLabel("Проект: новый проект\nДобавьте фото или задайте вопрос")
        self.context.setObjectName("assistantContext")
        self.context.setWordWrap(True)
        layout.addWidget(self.context)

        self.transcript = QScrollArea()
        self.transcript.setAcceptDrops(False)
        self.transcript.viewport().setAcceptDrops(False)
        self.transcript.setObjectName("assistantTranscript")
        self.transcript.setWidgetResizable(True)
        self.transcript.setFrameShape(QFrame.Shape.NoFrame)
        self.transcript.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.transcript.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._messages_widget = QWidget()
        self._messages_widget.setObjectName("assistantMessages")
        self._messages_layout = QVBoxLayout(self._messages_widget)
        self._messages_layout.setContentsMargins(3, 3, 3, 3)
        self._messages_layout.setSpacing(8)
        self._messages_layout.addStretch(1)
        self.transcript.setWidget(self._messages_widget)
        layout.addWidget(self.transcript, 1)
        self.append_message("assistant", "Добавьте фото, таблицу описаний или документ кнопкой «Добавить файл» либо перетащите сюда. Фото я проанализирую, а по загруженным материалам можно задавать вопросы здесь же.")

        self.photo_frame: QFrame | None = None
        self.photo_name: QLabel | None = None
        self.photo_preview: QLabel | None = None
        self.clear_photo_button: QPushButton | None = None
        self._photo_path: Path | None = None
        self._photo_base_pixmap: QPixmap | None = None
        self._photo_columns: list[dict] = []
        self._photo_image_size = (0, 0)

        self.composer = MessageComposer()
        self.composer.setAcceptDrops(False)
        self.composer.viewport().setAcceptDrops(False)
        self.composer.setObjectName("assistantComposer")
        self.composer.setPlaceholderText("Спросить о керне… Enter — отправить, Shift+Enter — новая строка")
        self.composer.setFixedHeight(82)
        self.composer.submitted.connect(self._send)
        layout.addWidget(self.composer)
        bottom = QHBoxLayout()
        self.attach_file_button = QPushButton("＋ Добавить файл")
        self.attach_file_button.setObjectName("assistantAction")
        self.attach_file_button.setToolTip("Добавить фото, Excel или документ; можно выбрать несколько файлов")
        self.attach_file_button.clicked.connect(self.files_requested)
        bottom.addWidget(self.attach_file_button)
        bottom.addStretch()
        self.send_button = QPushButton("➤  Отправить")
        self.send_button.setObjectName("assistantSend")
        self.send_button.clicked.connect(self._send)
        bottom.addWidget(self.send_button)
        layout.addLayout(bottom)

    def append_message(self, role: str, message: str, *, correction: dict | None = None) -> None:
        name = "Вы" if role == "user" else "ИИ-седиментолог" if role == "assistant" else "Система"
        color = "#e6f3ff" if role == "user" else "#f0f6fc" if role == "assistant" else "#fff4e3"
        bubble = QFrame(self._messages_widget)
        bubble.setStyleSheet(f"QFrame {{ background: {color}; border-radius: 11px; }}")
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(10, 9, 10, 9)
        bubble_layout.setSpacing(4)
        heading = QLabel(name, bubble)
        heading.setStyleSheet("color: #153572; font-weight: 700; background: transparent;")
        bubble_layout.addWidget(heading)
        body = QLabel(message, bubble)
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        body.setStyleSheet("color: #153572; background: transparent;")
        bubble_layout.addWidget(body)
        if role == "assistant" and correction:
            correction_button = QPushButton("✎ Исправить ответ", bubble)
            correction_button.setToolTip("Сохранить правильный ответ специалиста для следующих вопросов и дообучения")
            correction_button.clicked.connect(
                lambda _checked=False, data=dict(correction): self.correction_requested.emit(data)
            )
            bubble_layout.addWidget(correction_button)
        timestamp = QLabel(datetime.now().strftime("%H:%M"), bubble)
        timestamp.setStyleSheet("color: #6980a4; font-size: 10px; background: transparent;")
        bubble_layout.addWidget(timestamp)
        self._append_item(bubble)

    def append_photo_matches(self, hits: list[dict], note: str) -> None:
        """Show file matches as actions that open only real local photographs."""
        bubble = QFrame(self._messages_widget)
        bubble.setStyleSheet("QFrame { background: #f0f6fc; border-radius: 11px; }")
        layout = QVBoxLayout(bubble)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(6)
        title = QLabel("ИИ-седиментолог", bubble)
        title.setStyleSheet("color: #153572; font-weight: 700; background: transparent;")
        layout.addWidget(title)
        summary = QLabel(note, bubble)
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #153572; background: transparent;")
        layout.addWidget(summary)
        for index, hit in enumerate(hits, start=1):
            path = Path(str(hit.get("path") or ""))
            name = QLabel(f"{index}. {path.name}\n{path}", bubble)
            name.setTextFormat(Qt.TextFormat.PlainText)
            name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            name.setWordWrap(True)
            name.setStyleSheet("color: #153572; background: transparent;")
            layout.addWidget(name)
            actions = QHBoxLayout()
            show = QPushButton("Показать в чате", bubble)
            show.clicked.connect(lambda _checked=False, file_path=path: self._show_found_photo(file_path))
            actions.addWidget(show)
            open_file = QPushButton("Открыть файл", bubble)
            open_file.clicked.connect(lambda _checked=False, file_path=path: self._open_found_photo(file_path))
            actions.addWidget(open_file)
            layout.addLayout(actions)
        self._append_item(bubble)

    def _show_found_photo(self, path: Path) -> None:
        self.found_photo_requested.emit(str(path))

    def _open_found_photo(self, path: Path) -> None:
        if not path.is_file() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self.append_message("system", f"Не удалось открыть фото «{path.name}». Проверьте доступность файла.")

    def _append_item(self, widget: QWidget) -> None:
        scroll = self.transcript.verticalScrollBar()
        follow_latest = scroll.value() >= scroll.maximum() - 45
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, widget)
        if follow_latest:
            QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        scroll = self.transcript.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def _scroll_messages(self, direction: int) -> None:
        scroll = self.transcript.verticalScrollBar()
        scroll.setValue(scroll.value() + direction * max(140, self.transcript.viewport().height() * 3 // 4))

    def set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.attach_file_button.setEnabled(not busy)
        if self.clear_photo_button is not None:
            self.clear_photo_button.setEnabled(not busy)
        self.status.setText("●  Обрабатываю запрос…" if busy else "●  Готов к работе")

    def set_photo(self, path: str | Path) -> None:
        source = Path(path)
        reader = QImageReader(str(source))
        reader.setAutoTransform(False)
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > 900:
            factor = 900 / max(size.width(), size.height())
            reader.setScaledSize(QSize(max(1, round(size.width() * factor)), max(1, round(size.height() * factor))))
        image = reader.read()
        if image.isNull():
            raise ValueError(f"Не удалось открыть фото: {reader.errorString()}")
        if self.photo_frame is not None:
            if self.clear_photo_button is not None:
                self.clear_photo_button.hide()
            if self.photo_name is not None:
                self.photo_name.setText(self.photo_name.text() + " · прежнее вложение")
        self._photo_path = source
        self._photo_columns = []
        self._photo_image_size = (0, 0)
        self._photo_base_pixmap = QPixmap.fromImage(image)
        self.photo_frame = QFrame(self._messages_widget)
        self.photo_frame.setObjectName("assistantPhotoFrame")
        self.photo_frame.setStyleSheet("QFrame#assistantPhotoFrame { background: white; border: 1px solid #dceafb; border-radius: 11px; }")
        photo_layout = QVBoxLayout(self.photo_frame)
        photo_layout.setContentsMargins(9, 9, 9, 9)
        photo_layout.setSpacing(6)
        photo_header = QHBoxLayout()
        self.photo_name = QLabel(self.photo_frame)
        self.photo_name.setWordWrap(True)
        self.photo_name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        photo_header.addWidget(self.photo_name, 1)
        self.clear_photo_button = QPushButton("✕", self.photo_frame)
        self.clear_photo_button.setToolTip("Убрать фото из текущего контекста чата")
        self.clear_photo_button.clicked.connect(self.photo_clear_requested)
        photo_header.addWidget(self.clear_photo_button)
        photo_layout.addLayout(photo_header)
        self.photo_preview = PhotoPreviewLabel(self.photo_frame)
        self.photo_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_preview.setToolTip("Дважды щёлкните, чтобы открыть фото крупно")
        self.photo_preview.double_clicked.connect(self._open_photo_preview)
        photo_layout.addWidget(self.photo_preview)
        self.photo_name.setText(source.name)
        self._render_photo_preview()
        self._append_item(self.photo_frame)
        QTimer.singleShot(0, self._scroll_to_bottom)

    def show_photo_columns(self, columns: list[dict], image_width: int, image_height: int) -> None:
        if self._photo_base_pixmap is None or image_width <= 0 or image_height <= 0:
            return
        self._photo_columns = list(columns)
        self._photo_image_size = (image_width, image_height)
        self._render_photo_preview()

    def _render_photo_preview(self) -> None:
        if self._photo_base_pixmap is None or self.photo_preview is None:
            return
        available = max(150, self.transcript.viewport().width() - 36)
        preview_height = 460 if available < 500 else 650
        preview = self._photo_base_pixmap.scaled(
            min(600, available), preview_height,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        image_width, image_height = self._photo_image_size
        if image_width <= 0 or image_height <= 0:
            self.photo_preview.setPixmap(preview)
            return
        painter = QPainter(preview)
        painter.setPen(QPen(QColor("#00bfe8"), 2))
        painter.setBrush(QColor(0, 191, 232, 35))
        for number, box in enumerate(self._photo_columns, start=1):
            x = box["left"] * preview.width() / image_width
            y = box["top"] * preview.height() / image_height
            width = (box["right"] - box["left"]) * preview.width() / image_width
            height = (box["bottom"] - box["top"]) * preview.height() / image_height
            painter.drawRect(round(x), round(y), round(width), round(height))
            painter.drawText(round(x) + 4, round(y) + 17, str(number))
        painter.end()
        self.photo_preview.setPixmap(preview)

    def _open_photo_preview(self) -> None:
        if self._photo_path is None:
            return
        reader = QImageReader(str(self._photo_path))
        reader.setAutoTransform(False)
        size = reader.size()
        if size.isValid():
            factor = min(1.0, 850 / max(1, size.width()), 12000 / max(1, size.height()),
                         math.sqrt(12_000_000 / max(1, size.width() * size.height())))
            if factor < 1.0:
                reader.setScaledSize(QSize(max(1, round(size.width() * factor)), max(1, round(size.height() * factor))))
        image = reader.read()
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        width, height = self._photo_image_size
        if width > 0 and height > 0 and self._photo_columns:
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#00bfe8"), 3))
            painter.setBrush(QColor(0, 191, 232, 25))
            for number, box in enumerate(self._photo_columns, start=1):
                x = box["left"] * pixmap.width() / width
                y = box["top"] * pixmap.height() / height
                box_width = (box["right"] - box["left"]) * pixmap.width() / width
                box_height = (box["bottom"] - box["top"]) * pixmap.height() / height
                painter.drawRect(round(x), round(y), round(box_width), round(box_height))
                painter.drawText(round(x) + 5, round(y) + 22, str(number))
            painter.end()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Фото и столбики · {self._photo_path.name}")
        dialog.resize(920, 740)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(False)
        label = QLabel()
        label.setPixmap(pixmap)
        label.resize(pixmap.size())
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.exec()

    def clear_photo(self) -> None:
        if self.clear_photo_button is not None:
            self.clear_photo_button.hide()
        if self.photo_name is not None:
            self.photo_name.setText(self.photo_name.text() + " · откреплено")
        self._photo_path = None
        self._photo_base_pixmap = None
        self._photo_columns = []
        self._photo_image_size = (0, 0)
        self.photo_frame = None
        self.photo_name = None
        self.photo_preview = None
        self.clear_photo_button = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.expand_button.setText("Уже" if self.width() >= 500 else "Шире")
        if self._photo_base_pixmap is not None:
            QTimer.singleShot(0, self._render_photo_preview)

    def _send(self) -> None:
        if not self.send_button.isEnabled():
            return
        question = self.composer.toPlainText().strip()
        if question:
            self.composer.clear()
            self.send_requested.emit(question)
        elif self._photo_path is not None:
            self.photo_analysis_requested.emit()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
