"""Draggable edges of numbered core columns on a photo."""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsObject


class CoreColumnEdges(QGraphicsObject):
    changed = Signal(object)

    def __init__(self, area, image_size, columns, parent=None, *, verified=False, source_window=None, first_number=1):
        super().__init__(parent)
        self.area = QRectF(area)
        self.width, self.height = image_size
        self.on_screen_only = source_window is not None
        self.source_window = QRectF(source_window) if source_window is not None else QRectF(0, 0, self.width, self.height)
        self.first_number = first_number
        self.drag = None
        self.hover = None
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setZValue(5)
        self.set_columns(columns, verified)

    def set_columns(self, columns, verified=False):
        self.columns = [dict(box) for box in columns]
        self.verified = verified
        self.update()

    def boundingRect(self):
        return self.area

    def _rect(self, box):
        sx = self.area.width() / self.source_window.width()
        sy = self.area.height() / self.source_window.height()
        return QRectF(self.area.left() + (box["left"] - self.source_window.left()) * sx,
                      self.area.top() + (box["top"] - self.source_window.top()) * sy,
                      (box["right"] - box["left"]) * sx, (box["bottom"] - box["top"]) * sy)

    def _tolerance(self):
        scene = self.scene()
        views = scene.views() if scene is not None else []
        scale = abs(self.deviceTransform(views[0].viewportTransform()).m11()) if views else 1.0
        return 5 / max(.05, scale)

    def shape(self):
        edges = QPainterPath()
        for box in self.columns:
            edges.addRect(self._rect(box))
        stroker = QPainterPathStroker()
        stroker.setWidth(2 * self._tolerance())
        photo = QPainterPath()
        photo.addRect(self.area)
        return stroker.createStroke(edges).intersected(photo)

    def _edge_at(self, point):
        tolerance = self._tolerance()
        best = None
        for index, box in enumerate(self.columns):
            rect = self._rect(box)
            if not rect.adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(point):
                continue
            side = "left" if abs(point.x() - rect.left()) < abs(point.x() - rect.right()) else "right"
            end = "top" if abs(point.y() - rect.top()) < abs(point.y() - rect.bottom()) else "bottom"
            x, y = getattr(rect, side)(), getattr(rect, end)()
            # An internal crop seam can sit close to an edge outside this view.
            # Only let the user pick source edges that are actually visible.
            dx = abs(point.x() - x) if self.area.left() - 1e-6 <= x <= self.area.right() + 1e-6 else float("inf")
            dy = abs(point.y() - y) if self.area.top() - 1e-6 <= y <= self.area.bottom() + 1e-6 else float("inf")
            if min(dx, dy) <= tolerance:
                candidate = (min(dx, dy), index, side if dx <= tolerance else "", end if dy <= tolerance else "")
                if best is None or candidate < best:
                    best = candidate
        return best[1:] if best else None

    def hoverMoveEvent(self, event):
        self.hover = self._edge_at(event.pos())
        if self.hover:
            _, side, end = self.hover
            if side and end:
                cursor = Qt.CursorShape.SizeFDiagCursor if (side == "left") == (end == "top") else Qt.CursorShape.SizeBDiagCursor
            else:
                cursor = Qt.CursorShape.SizeHorCursor if side else Qt.CursorShape.SizeVerCursor
            self.setCursor(cursor)
        else:
            self.unsetCursor()
        self.update()

    def hoverLeaveEvent(self, event):
        self.hover = None
        self.unsetCursor()
        self.update()

    def mousePressEvent(self, event):
        hit = self._edge_at(event.pos())
        if not hit:
            event.ignore()
            return
        index, side, end = hit
        self.drag = (index, side, end, event.pos(), dict(self.columns[index]))
        event.accept()

    def mouseMoveEvent(self, event):
        if self.drag is None:
            return
        index, side, end, start, original = self.drag
        delta = event.pos() - start
        box = dict(original)
        if side:
            low, high = (0, box["right"] - 2) if side == "left" else (box["left"] + 2, self.width)
            box[side] = max(low, min(high, original[side] + delta.x() * self.source_window.width() / self.area.width()))
        if end:
            low, high = (0, box["bottom"] - 2) if end == "top" else (box["top"] + 2, self.height)
            box[end] = max(low, min(high, original[end] + delta.y() * self.source_window.height() / self.area.height()))
        self.columns[index] = box
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.drag is None:
            return
        self.mouseMoveEvent(event)
        index, _, _, _, original = self.drag
        self.drag = None
        if self.columns[index] != original:
            self.changed.emit([dict(box) for box in self.columns])

    def paint(self, painter, option, widget=None):
        if self.on_screen_only and widget is None:
            return  # The assembled view's resize handles are interactive controls.
        painter.save()
        painter.setClipRect(self.area, Qt.ClipOperation.IntersectClip)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index, box in enumerate(self.columns):
            active = self.drag or self.hover
            pen = QPen(QColor("#087fca" if active and active[0] == index else "#12b8d9"), 2)
            pen.setCosmetic(True)
            if not self.verified and not (active and active[0] == index):
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            rect = self._rect(box)
            painter.drawRect(rect)
            painter.drawText(QRectF(rect.left(), rect.top(), max(12, rect.width()), 14),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, str(self.first_number + index))
        painter.restore()
