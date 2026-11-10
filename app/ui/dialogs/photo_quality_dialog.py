"""Inspect photographic warnings and the exact candidate label exclusions."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QVBoxLayout)
from app.infrastructure.photo_quality import MAX_EXCLUSION_POINTS, MAX_MANUAL_EXCLUSIONS, normalize_manual_exclusions
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


def exclusion_path(area, width, height):
    """Project a manual exclusion to the original photo coordinates."""
    path = QPainterPath()
    if "rect" in area:
        l, t, r, b = area["rect"]
        path.addRect(QRectF(l * width, t * height, (r - l) * width, (b - t) * height))
        return path
    points = area["points"]
    x, y = points[0]
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
        self.exclusions = normalize_manual_exclusions(exclusions)
        self.marks = [self._add_mark(area) for area in self.exclusions]
        self.tool = None
        self.brush_size = 18
        self.show_marks = True
        self._anchor = None
        self._draft = None
        self._draft_mark = None

    def _add_mark(self, area):
        width, height = self.photo.width(), self.photo.height()
        colour = QColor(224, 55, 80, 110)
        path = exclusion_path(area, width, height)
        if "rect" in area:
            mark = self.scene_data.addPath(path, QPen(Qt.PenStyle.NoPen), QBrush(colour))
        else:
            pen = QPen(colour, 2 * area["radius"] * min(width, height))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            mark = self.scene_data.addPath(path, pen)
        mark.setZValue(4)
        mark.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        return mark

    def _point(self, point):
        return [round(max(0., min(1., point.x() / self.photo.width())), 5),
                round(max(0., min(1., point.y() / self.photo.height())), 5)]

    def set_tool(self, tool):
        if tool == self.tool:
            return
        self._finish_exclusion(keep=False)
        self.tool = tool
        self.setDragMode(self.DragMode.NoDrag if tool else self.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if tool else Qt.CursorShape.OpenHandCursor)

    def set_marks_visible(self, visible):
        self.show_marks = visible
        for mark in self.marks:
            mark.setVisible(visible)
        if self._draft_mark is not None:
            self._draft_mark.setVisible(visible)

    def mousePressEvent(self, event):
        if self.tool and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point) and len(self.exclusions) < MAX_MANUAL_EXCLUSIONS:
                self._anchor = self._point(point)
                if self.tool == "rect":
                    self._draft = {"rect": self._anchor * 2}
                else:
                    radius = self.brush_size / (2 * self.transform().m11() * min(self.photo.width(), self.photo.height()))
                    self._draft = {"radius": round(min(.25, max(.00001, radius)), 6), "points": [self._anchor]}
                self._draft_mark = self._add_mark(self._draft)
                self._draft_mark.setVisible(self.show_marks)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._draft is not None:
            point = self._point(self.mapToScene(event.position().toPoint()))
            if "rect" in self._draft:
                x, y = self._anchor
                self._draft["rect"] = [min(x, point[0]), min(y, point[1]), max(x, point[0]), max(y, point[1])]
            else:
                points = self._draft["points"]
                if point != points[-1]:
                    if len(points) == MAX_EXCLUSION_POINTS:
                        points[:] = points[::2]
                    points.append(point)
            self._draft_mark.setPath(exclusion_path(self._draft, self.photo.width(), self.photo.height()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.tool and event.button() == Qt.MouseButton.LeftButton:
            if self._draft is not None:
                self.mouseMoveEvent(event)
                self._finish_exclusion(keep=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _finish_exclusion(self, *, keep):
        if self._draft_mark is None:
            return
        valid = normalize_manual_exclusions([self._draft]) if keep else []
        if valid:
            self.exclusions.append(valid[0])
            self.marks.append(self._draft_mark)
        else:
            self.scene_data.removeItem(self._draft_mark)
        self._draft_mark = self._draft = self._anchor = None

    def undo_exclusion(self):
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
        info = QLabel(f'Автоматически выделено этикеток: {len(self.boxes)}. Оранжевые области — автоисключения, красные — ручные.')
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
            tools.addWidget(brush)
            size = QSpinBox()
            size.setRange(4, 100)
            size.setValue(self.view.brush_size)
            size.setSuffix(" пикс.")
            size.setToolTip("Размер кисти")
            size.setMaximumWidth(105)
            size.valueChanged.connect(lambda value: setattr(self.view, "brush_size", value))
            tools.addWidget(size)
            rectangle = QPushButton("Прямоугольник")
            rectangle.setCheckable(True)
            rectangle.setToolTip("Выделить прямоугольную область для исключения из анализа")
            tools.addWidget(rectangle)

            def choose_tool(tool, other, active):
                if active:
                    other.setChecked(False)
                    toggle.setChecked(True)
                self.view.set_tool(tool if active else None)
                if self.column_edges is not None:
                    self.column_edges.setEnabled(not active)

            brush.toggled.connect(lambda active: choose_tool("brush", rectangle, active))
            rectangle.toggled.connect(lambda active: choose_tool("rect", brush, active))
            undo = QPushButton("Отменить выделение")
            undo.clicked.connect(self.view.undo_exclusion)
            tools.addWidget(undo)
            clear = QPushButton("Убрать ручные исключения")
            clear.clicked.connect(self._clear_exclusions)
            tools.addWidget(clear)
            layout.addLayout(tools)
            hint = QLabel("Кисть закрашивает область. Для прямоугольника протяните от одного угла к другому. "
                          "Выключите инструмент, чтобы двигать голубые рамки. Колесо мыши меняет масштаб.")
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
            self.view.undo_exclusion()

    def _show_exclusions(self, visible):
        self.view.set_marks_visible(visible)
        for item in self.boxes:
            item.setVisible(visible)

    def edited_columns(self):
        return self.column_edges.columns if self.column_edges is not None else None

    def showEvent(self,event):
        super().showEvent(event)
        if self.focus and not self.focus.isEmpty():
            self.view.fitInView(self.focus, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.view.fit_photo()
