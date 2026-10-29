"""Core descriptions stay reachable during processing, including whole columns."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from types import SimpleNamespace
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtTest import QTest
from app.domain.models import FaciesDetection, PhotoRecord
from app.ui.widgets import workspace_canvas as ui

_APP = QApplication.instance() or QApplication([])


class CoreHoverUpdatesTests(unittest.TestCase):
    def setUp(self):
        pixmap = QPixmap(40, 100)
        pixmap.fill(Qt.GlobalColor.gray)
        self.detection = FaciesDetection("Не определена", 0, [
            QPointF(2, 2), QPointF(35, 2), QPointF(35, 96), QPointF(2, 96)],
            {"__core_column_description": "1", "__interpretation_status": "waiting"})
        self.record = PhotoRecord("one", "memory.png", pixmap, [self.detection])
        self.canvas = ui.WorkspaceCanvas()
        self.canvas.add_photos([self.record])
        self.canvas.show_stack([self.record], description_only=True)
        self.addCleanup(self.canvas.close)
        self.addCleanup(ui.hide_facies_confidence_popup)

    @staticmethod
    def popup_text():
        _APP.processEvents()
        return "\n".join(label.text() for label in ui._confidence_popup.findChildren(QLabel)
                         if not label.isHidden())

    def test_hover_on_assembled_whole_core_shows_waiting_status(self):
        stack = self.canvas._stack_item
        _, _, _, rect, overlay = stack._placements[0]
        self.assertEqual(stack._column_at(rect.center())[2], "core")
        self.assertEqual(overlay._detection_at(rect.center(), include_core_columns=True), 0)
        # Whole-column hover must not turn the neutral frame into an editable
        # facies mask or prevent dragging the source photo.
        self.assertEqual(overlay._detection_at(rect.center()), -1)
        overlay.hoverMoveEvent(SimpleNamespace(pos=rect.center, screenPos=lambda: QPoint(20, 20)))
        self.assertTrue(ui._confidence_popup.isVisible())
        self.assertIn("ещё готовится", self.popup_text())

    def test_completed_description_updates_hover_without_rebuilding_stack(self):
        stack = self.canvas._stack_item
        _, _, _, rect, overlay = stack._placements[0]
        overlay.hoverMoveEvent(SimpleNamespace(pos=rect.center, screenPos=lambda: QPoint(20, 20)))
        self.detection.label = "Предположительно: слоистая фация"
        self.detection.attributes = {
            **self.detection.attributes, "__interpretation_status": "draft",
            "Краткое описание": "Серый керн с тонкой волнистой слоистостью.",
            "Название породы": "Предположительно: алевролит", "Текстура": "Слоистая",
        }
        old_position = QPointF(stack.pos())
        self.canvas.update_photo_detections(self.record)
        self.assertIs(self.canvas._stack_item, stack)
        self.assertEqual(stack.pos(), old_position)
        self.assertIs(overlay.hover_detections[0], self.detection)
        self.assertIn("Серый керн", self.popup_text())
        self.assertIn("алевролит", self.popup_text())
        self.assertNotIn("ещё готовится", self.popup_text())
        self.assertEqual(overlay.detections[0].label, self.detection.label)

    def test_real_facies_has_hover_priority_over_whole_column(self):
        mask = FaciesDetection("A", .8, [QPointF(4, 4), QPointF(20, 4), QPointF(20, 30), QPointF(4, 30)])
        overlay = ui.FaciesOverlayItem(self.record.pixmap.size(), QRectF(self.record.pixmap.rect()),
                                      [mask, self.detection])
        self.assertEqual(overlay._detection_at(QPointF(10, 10), include_core_columns=True), 0)

    def test_view_delivers_hover_to_unpainted_column_overlay(self):
        self.canvas.resize(900, 600)
        self.canvas.show()
        self.canvas.fit_content()
        _APP.processEvents()
        stack = self.canvas._stack_item
        _, _, _, rect, overlay = stack._placements[0]
        point = self.canvas.mapFromScene(overlay.mapToScene(rect.center()))
        QTest.mouseMove(self.canvas.viewport(), point)
        QTest.mouseMove(self.canvas.viewport(), point + QPoint(1, 1), 10)
        _APP.processEvents()
        self.assertTrue(ui._confidence_popup is not None and ui._confidence_popup.isVisible())
        self.assertIn("Описание столбика", self.popup_text())

    def test_click_on_assembled_whole_column_selects_original_for_manual_form(self):
        selected = []
        self.canvas.facies_selected.connect(lambda record, detection: selected.append((record, detection)))
        self.canvas.resize(900, 600)
        self.canvas.show()
        self.canvas.fit_content()
        _APP.processEvents()
        _, _, _, rect, overlay = self.canvas._stack_item._placements[0]
        point = self.canvas.mapFromScene(overlay.mapToScene(rect.center()))
        QTest.mouseClick(self.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0][0], self.record)
        self.assertIs(selected[0][1], self.detection)


if __name__ == "__main__":
    unittest.main()
