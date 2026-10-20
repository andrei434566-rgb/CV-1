"""Non-destructive preview of curved visual contacts in core photographs."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from app.infrastructure.excel_core_description import create_automatic_interval_detections
from app.infrastructure.visual_contact_audit import audit_visual_contacts, is_uv_photo


class VisualContactReviewDialog(QDialog):
    """Inspect draft polygons and contact curves before explicitly applying them."""

    apply_requested = Signal(object, object, object)

    def __init__(self, records, depth_getter, parent=None, selected_record=None):
        super().__init__(parent)
        self._records = [
            record for record in records
            if not is_uv_photo(record.path) and not record.pixmap.isNull()
        ]
        self._depth_getter = depth_getter
        self._drafts = []
        self._segments = []
        self._audit = {}
        self.resize(1050, 850)
        self.setWindowTitle("Переходы между фациями — проверка контактов")

        layout = QVBoxLayout(self)
        info = QLabel(
            "Предварительные границы по изменению текстуры. Неровные линии показаны "
            "как гипотезы; названия фаций не назначаются. Ни одна маска не изменится "
            "без вашего подтверждения. UV-фотографии здесь исключены."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.selector = QComboBox()
        for record in self._records:
            self.selector.addItem(f"{record.well_name} · {Path(record.path).name}")
        self.selector.currentIndexChanged.connect(self._refresh)
        layout.addWidget(self.selector)

        self.status = QLabel("Выберите фотографию.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setStyleSheet("background: #171f29; border-radius: 6px;")
        layout.addWidget(self.preview, 1)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Столбик", "Граница, м", "Y, пикс.", "Неровность, пикс.",
            "Геометрия", "Статус",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(170)
        layout.addWidget(self.table)

        controls = QHBoxLayout()
        save = QPushButton("Экспорт контактов (JSON)…")
        save.clicked.connect(self._export)
        controls.addWidget(save)
        controls.addStretch()
        apply = QPushButton("Создать редактируемые черновики")
        apply.clicked.connect(self._apply)
        controls.addWidget(apply)
        cancel = QPushButton("Закрыть без изменений")
        cancel.clicked.connect(self.reject)
        controls.addWidget(cancel)
        layout.addLayout(controls)

        if not self._records:
            self.status.setText("Нет обычных фотографий керна для проверки.")
            self.selector.setEnabled(False)
            save.setEnabled(False)
            apply.setEnabled(False)
        else:
            if selected_record is not None:
                for i, record in enumerate(self._records):
                    if record is selected_record:
                        self.selector.setCurrentIndex(i)
                        break
            QTimer.singleShot(0, self._refresh)

    def _refresh(self, *_args):
        index = self.selector.currentIndex()
        if index < 0 or index >= len(self._records):
            return
        record = self._records[index]
        self.table.setRowCount(0)
        self._audit, self._drafts, self._segments = {}, [], []
        self.preview.clear()
        self.status.setText(f"Анализ: {Path(record.path).name}…")
        try:
            top, base = self._depth_getter(record)
            # Never edit the source record during preview. Prefer already
            # identified physical columns instead of silently redetecting them.
            draft, segments = create_automatic_interval_detections(
                Path(record.path), record.pixmap, top, base,
                core_columns=record.core_columns if record.core_columns else None,
                label="Новый контур",
            )
            audit = audit_visual_contacts(draft, source_photo=Path(record.path).name)
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.setText(f"Ошибка определения контактов: {exc}")
            return
        except Exception as exc:
            self.status.setText(f"Не удалось обработать фото: {type(exc).__name__}: {exc}")
            return
        self._drafts, self._segments, self._audit = draft, segments, audit
        self.status.setText(
            f"Столбиков: {audit['column_count']} · черновых интервалов: "
            f"{audit['interval_count']} · предполагаемых переходов: {audit['contact_count']}. "
            "Это не подтверждённые геологические контакты."
        )
        contacts = audit["contacts"]
        self.table.setRowCount(len(contacts))
        for row, contact in enumerate(contacts):
            values = [
                contact["column"],
                f"{contact['depth_m']:.3f}" if contact["depth_m"] is not None else "нет привязки",
                contact["pixel_y"],
                contact["amplitude_px"],
                "неровная" if contact["geometry"] == "curved" else "близка к прямой",
                "нужна проверка" + (" · проверьте глубины" if contact["depth_conflict"] else ""),
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self._draw_preview(record, draft, contacts)

    def _draw_preview(self, record, drafts, contacts):
        shown = record.pixmap.scaled(
            1000, 480, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if shown.isNull():
            return
        canvas = QPixmap(shown)
        sx = canvas.width() / max(1, record.pixmap.width())
        sy = canvas.height() / max(1, record.pixmap.height())
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#E9BD58"), 1.2))
        for draft in drafts:
            poly = QPolygonF([QPointF(p.x() * sx, p.y() * sy) for p in draft.polygon])
            painter.drawPolygon(poly)
        painter.setPen(QPen(QColor("#E35454"), 2.6))
        for contact in contacts:
            trace = contact["trace"]
            if len(trace) > 1:
                painter.drawPolyline(QPolygonF([QPointF(x * sx, y * sy) for x, y in trace]))
        painter.end()
        self.preview.setPixmap(canvas)
        self.preview.setToolTip("Жёлтый: предложенные интервалы. Красный: предполагаемые контакты.")

    def _export(self):
        if not self._audit:
            QMessageBox.information(self, "Контакты", "Сначала выберите и обработайте фото.")
            return
        name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить контакты", "visual_contacts_review.json", "JSON (*.json)"
        )
        if name:
            try:
                Path(name).write_text(json.dumps(self._audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                QMessageBox.critical(self, "Экспорт", str(exc))
            else:
                QMessageBox.information(self, "Экспорт", "Предварительные контакты сохранены. Для обучения требуется проверка геологом.")

    def _apply(self):
        index = self.selector.currentIndex()
        if index < 0 or index >= len(self._records) or not self._drafts:
            QMessageBox.information(self, "Контакты", "Для фото пока нет редактируемых интервалов.")
            return
        if QMessageBox.question(
            self, "Заменить маски на этом фото?",
            "Все текущие маски ВЫБРАННОЙ фотографии будут заменены новыми черновыми "
            "интервалами с неровными границами. Названия фаций и подтверждения обучения "
            "НЕ переносятся. Сохраните проект перед заменой, если нужны старые маски.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        # Training approval is NEVER carried over from prior human masks.
        for draft in self._drafts:
            draft.training_ready = False
        self.apply_requested.emit(self._records[index], list(self._drafts), list(self._segments))
        self.accept()
