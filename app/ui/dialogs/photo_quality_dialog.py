"""Inspect photographic warnings and the exact candidate label exclusions."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QVBoxLayout)
from app.infrastructure.photo_quality import MAX_EXCLUSION_POINTS, MAX_EXCLUSION_STROKES, normalize_exclusion_strokes
from app.ui.dialogs.feature_evidence_dialog import EvidenceView
from app.ui.widgets.core_column_edges import CoreColumnEdges

REVIEW_STYLE='''
QDialog { background: #f7faff; }
QLabel,QCheckBox { color: #163660; }
QLineEdit,QComboBox,QPlainTextEdit,QTableWidget,QSpinBox { background: white; color: #163660; border: 1px solid #ccdaed; border-radius: 4px; padding: 4px; }
QHeaderView::section { background: #eaf2fc; color: #163660; padding: 5px; border: 1px solid #d7e3f2; }
QPushButton { color: #164c85; background: white; border: 1px solid #c8daf0; border-radius: 5px; padding: 6px 10px; }
QPushButton:hover { background: #eaf4ff; }
QPushButton:checked { background: #dceeff; border-color: #087fca; }
'''


def exclusion_path(stroke, width, height):
    """Draw the same round strokes on zoomed previews and colour crops."""
    points = stroke["points"]
    x, y = points[0]
    path = QPainterPath()
    path.moveTo(x * width, y * height)
    if len(points) == 1:
        path.lineTo(x * width + .001, y * height)
    else:
        for x, y in points[1:]:
            path.lineTo(x * width, y * height)
    return path


class _ExclusionView(EvidenceView):
    def __init__(self, pixmap, exclusions):
        super().__init__(pixmap)
        self.selection_mode = False
        self.setDragMode(self.DragMode.ScrollHandDrag)
        self.exclusions = normalize_exclusion_strokes(exclusions)
        self.marks = [self._add_mark(stroke) for stroke in self.exclusions]
        self.brush = False
        self.brush_size = 18
        self.painting = False
        self.show_marks = True

    def _add_mark(self, stroke):
        width, height = self.photo.width(), self.photo.height()
        pen = QPen(QColor(224, 55, 80, 110), 2 * stroke["radius"] * min(width, height))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        mark = self.scene_data.addPath(exclusion_path(stroke, width, height), pen)
        mark.setZValue(4)
        mark.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        return mark

    def _point(self, point):
        return [round(max(0., min(1., point.x() / self.photo.width())), 5),
                round(max(0., min(1., point.y() / self.photo.height())), 5)]

    def set_brush(self, active):
        self.brush = active
        self.setDragMode(self.DragMode.NoDrag if active else self.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):
        if self.brush and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point) and len(self.exclusions) < MAX_EXCLUSION_STROKES:
                radius = self.brush_size / (2 * self.transform().m11() * min(self.photo.width(), self.photo.height()))
                stroke = {"radius": round(min(.25, max(.00001, radius)), 6), "points": [self._point(point)]}
                self.exclusions.append(stroke)
                self.marks.append(self._add_mark(stroke))
                self.marks[-1].setVisible(self.show_marks)
                self.painting = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.painting:
            stroke = self.exclusions[-1]
            point = self._point(self.mapToScene(event.position().toPoint()))
            if point != stroke["points"][-1]:
                if len(stroke["points"]) == MAX_EXCLUSION_POINTS:
                    stroke["points"][:] = stroke["points"][::2]
                stroke["points"].append(point)
                self.marks[-1].setPath(exclusion_path(stroke, self.photo.width(), self.photo.height()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.painting and event.button() == Qt.MouseButton.LeftButton:
            self.mouseMoveEvent(event)
            self.painting = False
            return
        super().mouseReleaseEvent(event)

    def undo_stroke(self):
        if self.exclusions:
            self.exclusions.pop()
            self.scene_data.removeItem(self.marks.pop())


class PhotoQualityDialog(QDialog):
    def __init__(self, pixmap, report, parent=None, *, columns=None, exclusions=None, focus=None):
        super().__init__(parent);self.resize(950,800);self.setWindowTitle('Качество фото и исключения')
        self.setStyleSheet(REVIEW_STYLE)
        self.focus = focus
        editable = exclusions is not None
        layout=QVBoxLayout(self)
        issues=[str(row.get('message','')) for row in report.get('issues',[]) if isinstance(row,dict)]
        note=QLabel('\n'.join(issues) or ('Фотографических предупреждений не найдено.' if report else 'Качество фото ещё не проверено.'))
        note.setWordWrap(True);layout.addWidget(note)
        self.view = _ExclusionView(pixmap, exclusions)
        layout.addWidget(self.view, 1)
        self.column_edges = None
        if columns:
            self.column_edges = CoreColumnEdges(self.view.sceneRect(), (pixmap.width(), pixmap.height()), columns)
            self.view.scene_data.addItem(self.column_edges)
        self.boxes=[];pen=QPen(QColor('#e38a09'),2);pen.setCosmetic(True)
        for box in report.get('label_boxes',[]):
            l,t,r,b=box
            item = self.view.scene_data.addRect(QRectF(l*pixmap.width(),t*pixmap.height(),(r-l)*pixmap.width(),(b-t)*pixmap.height()),pen)
            item.setZValue(3)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.boxes.append(item)
        info = QLabel(f'Автоматически выделено этикеток: {len(self.boxes)}. Оранжевые рамки — автоисключения, красные мазки — ручные.')
        info.setWordWrap(True);layout.addWidget(info)
        controls=QHBoxLayout()
        toggle=QCheckBox('Показать области исключения');toggle.setChecked(True)
        toggle.toggled.connect(self._show_exclusions);controls.addWidget(toggle)
        fit=QPushButton('Показать целиком');fit.clicked.connect(self.view.fit_photo);controls.addWidget(fit)
        layout.addLayout(controls)
        if editable:
            tools = QHBoxLayout()
            brush = QPushButton("Кисть исключения")
            brush.setCheckable(True)
            brush.toggled.connect(self.view.set_brush)
            brush.toggled.connect(lambda active: toggle.setChecked(True) if active else None)
            tools.addWidget(brush)
            size = QSpinBox()
            size.setRange(4, 100)
            size.setValue(self.view.brush_size)
            size.setSuffix(" пикс.")
            size.valueChanged.connect(lambda value: setattr(self.view, "brush_size", value))
            tools.addWidget(size)
            undo = QPushButton("Отменить мазок")
            undo.clicked.connect(self.view.undo_stroke)
            tools.addWidget(undo)
            clear = QPushButton("Убрать ручные исключения")
            clear.clicked.connect(self._clear_exclusions)
            tools.addWidget(clear)
            layout.addLayout(tools)
            hint = QLabel("Потяните за край голубой рамки. Колесо мыши меняет масштаб. При включённой кисти левая кнопка рисует исключения.")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
            buttons.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
            buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)
        else:
            close = QPushButton('Закрыть')
            close.clicked.connect(self.close)
            controls.addWidget(close)

    def _clear_exclusions(self):
        while self.view.exclusions:
            self.view.undo_stroke()

    def _show_exclusions(self, visible):
        self.view.show_marks = visible
        for item in self.boxes + self.view.marks:
            item.setVisible(visible)

    def edited_columns(self):
        return self.column_edges.columns if self.column_edges is not None else None

    def showEvent(self,event):
        super().showEvent(event)
        if self.focus and not self.focus.isEmpty():
            self.view.fitInView(self.focus, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.view.fit_photo()
