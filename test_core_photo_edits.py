import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.cv1_analysis import IntervalJob, write_interval_crop
from app.infrastructure.project_storage import load_project, save_project
from app.ui.dialogs.facies_dialog import make_interval_preview
from app.ui.dialogs.photo_quality_dialog import PhotoQualityDialog
from app.ui.widgets.workspace_canvas import WorkspaceCanvas


class CorePhotoEditsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.pixmap = QPixmap(300, 400)
        self.pixmap.fill(QColor("#888888"))
        self.columns = [{"left": 40, "top": 20, "right": 100, "bottom": 370},
                        {"left": 130, "top": 20, "right": 190, "bottom": 370}]

    def drag(self, view, start, end):
        first, last = (view.mapFromScene(point) for point in (start, end))
        QTest.mouseMove(view.viewport(), first)
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=first)
        QTest.mouseMove(view.viewport(), last)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=last)
        self.app.processEvents()

    def close_widget(self, widget):
        widget.close()
        widget.deleteLater()
        self.app.sendPostedEvents(widget, QEvent.Type.DeferredDelete)

    def test_board_drags_only_the_picked_edge(self):
        board = WorkspaceCanvas()
        self.addCleanup(self.close_widget, board)
        board.resize(800, 600)
        record = PhotoRecord("photo", "sample.png", self.pixmap, [], core_columns=self.columns)
        board.add_photos([record])
        board.show()
        board.focus_photos()
        self.app.processEvents()
        photo = board._items_by_identifier[record.identifier]
        area = photo.preview_rect
        start = photo.mapToScene(QPointF(area.left() + 100 * area.width() / 300, area.center().y()))
        changed = []
        board.core_columns_changed.connect(lambda item, boxes: changed.append((item, boxes)))
        self.drag(board, start, start + QPointF(12, 0))
        self.assertEqual(len(changed), 1)
        self.assertIs(changed[0][0], record)
        first, second = changed[0][1]
        self.assertGreater(first.pop("right"), 115)
        self.assertEqual(first, {"left": 40, "top": 20, "bottom": 370})
        self.assertEqual(second, self.columns[1])
        self.assertEqual(photo.pos(), QPointF())

    def test_large_preview_supports_resizing_and_both_exclusion_tools(self):
        dialog = PhotoQualityDialog(self.pixmap, {"label_boxes": [[.2, .2, .25, .25]]},
                                    columns=self.columns, exclusions=[])
        self.addCleanup(self.close_widget, dialog)
        dialog.show()
        self.app.processEvents()
        self.drag(dialog.view, QPointF(70, 20), QPointF(70, 42))
        self.assertAlmostEqual(dialog.edited_columns()[0]["top"], 42, delta=1)
        buttons = {button.text(): button for button in dialog.findChildren(QPushButton)}
        brush, rectangle = (buttons[name] for name in ("Кисть исключения", "Прямоугольник"))
        brush.click()
        self.drag(dialog.view, QPointF(65, 80), QPointF(90, 90))
        self.assertEqual(len(dialog.boxes), 1)
        self.assertEqual(len(dialog.view.exclusions), 1)
        self.assertGreaterEqual(len(dialog.view.exclusions[0]["points"]), 2)
        self.assertAlmostEqual(dialog.edited_columns()[0]["top"], 42, delta=1)
        rectangle.click()
        self.assertFalse(brush.isChecked())
        self.assertTrue(rectangle.isChecked())
        self.drag(dialog.view, QPointF(90, 180), QPointF(65, 135))
        self.assertEqual(len(dialog.view.exclusions), 2)
        bounds = dialog.view.exclusions[-1]["rect"]
        for actual, expected in zip(bounds, (65 / 300, 135 / 400, 90 / 300, 180 / 400)):
            self.assertAlmostEqual(actual, expected, delta=.003)
        self.drag(dialog.view, QPointF(70, 210), QPointF(70, 210))
        self.assertEqual(len(dialog.view.exclusions), 2)
        buttons["Отменить выделение"].click()
        self.assertIn("points", dialog.view.exclusions[0])
        buttons["Отменить выделение"].click()
        self.assertEqual(dialog.view.exclusions, [])
        self.assertTrue(dialog.boxes[0].isVisible())

    def test_assembled_frame_resizes_the_original_column_from_a_scaled_crop(self):
        board = WorkspaceCanvas()
        self.addCleanup(self.close_widget, board)
        board.resize(800, 600)
        first = FaciesDetection("Песчаник", .9, [QPointF(45, 10), QPointF(95, 10),
                                                QPointF(95, 170), QPointF(45, 170)], depth_from=0., depth_to=10.)
        second = FaciesDetection("Алевролит", .9, [QPointF(45, 170), QPointF(95, 170),
                                                  QPointF(95, 385), QPointF(45, 385)], depth_from=10., depth_to=20.)
        record = PhotoRecord("photo", "sample.png", self.pixmap, [first, second], core_columns=self.columns)
        board.show_correlation({"Р-31": [record]}, {"Р-31": ({"depth_range": (0., 40.)}, None)},
                               description_only=True)
        board.show()
        self.app.processEvents()
        stack = board._correlation_items[0]
        top, base = (placement[3] for placement in stack._placements)
        board.centerOn(stack.mapToScene(top.center()))
        changed = []
        board.core_columns_changed.connect(lambda item, boxes: changed.append((item, boxes)))
        seam = stack.mapToScene(QPointF(top.center().x(), base.top()))
        self.drag(board, seam, seam + QPointF(0, 20))
        self.assertEqual(changed, [])
        edge = stack.mapToScene(QPointF(top.center().x(), top.top()))
        self.drag(board, edge, edge + QPointF(0, 20))
        self.assertEqual(len(changed), 1)
        self.assertIs(changed[0][0], record)
        expected = 20 + 20 * (170 - 20) / top.height()
        self.assertAlmostEqual(changed[0][1][0]["top"], expected, delta=1.)
        self.assertEqual(changed[0][1][1], self.columns[1])
        self.assertEqual(first.polygon[0], QPointF(45, 10))
        record.core_columns = changed[0][1]
        record.core_columns_edited = True
        board.show_correlation({"Р-31": [record]}, {"Р-31": ({"depth_range": (0., 40.)}, None)},
                               description_only=True)
        rebuilt = board._correlation_items[0]
        new_top = rebuilt._placements[0][3]
        self.assertGreater(new_top.top(), top.top() + 20)
        self.assertAlmostEqual(new_top.bottom(), top.bottom())
        self.assertEqual(first.depth_from, 0.)

    def test_board_reaches_deep_well_end_and_expands_during_pan(self):
        board = WorkspaceCanvas()
        self.addCleanup(self.close_widget, board)
        board.resize(800, 600)
        board.show()
        self.app.processEvents()
        row = FaciesDetection("Алевролит", .9, [QPointF(40, 20), QPointF(100, 20),
                                              QPointF(100, 370), QPointF(40, 370)],
                              {"__core_column_description": "1", "__core_column_index": "0"},
                              depth_from=79_950., depth_to=79_990.)
        record = PhotoRecord("photo", "sample.png", self.pixmap, [row], core_columns=self.columns)
        board.show_stack([record], gis_preview={"depth_range": (0., 80_000.)}, description_only=True)
        stack = board._stack_item
        well = stack.sceneBoundingRect()
        self.assertGreater(well.bottom(), 30_000)
        self.assertTrue(board.sceneRect().contains(well))
        board.centerOn(well.center().x(), well.bottom())
        self.assertTrue(board.viewport().rect().contains(board.mapFromScene(well.center().x(), well.bottom())))
        crop = stack.mapToScene(stack._placements[0][3].center())
        board.centerOn(crop)
        self.app.processEvents()
        screenshot = board.viewport().grab().toImage()
        point = board.mapFromScene(crop) * screenshot.devicePixelRatio()
        self.assertEqual(screenshot.pixelColor(point), QColor("#888888"))
        rect = stack._placements[0][3]
        edge = board.mapFromScene(stack.mapToScene(QPointF(rect.left(), rect.center().y())))
        edge *= screenshot.devicePixelRatio()
        colours = [screenshot.pixelColor(edge + QPoint(dx, dy)) for dx in range(-1, 3) for dy in range(-8, 9)]
        self.assertTrue(any(colour.green() > colour.red() + 30 and colour.blue() > colour.green() for colour in colours))
        bar = board.verticalScrollBar()
        old_end = board.sceneRect().bottom()
        bar.setValue(bar.maximum())
        before = bar.value()
        start = board.viewport().rect().center()
        QTest.mousePress(board.viewport(), Qt.MouseButton.MiddleButton, pos=start)
        QTest.mouseMove(board.viewport(), start - QPoint(0, 70))
        QTest.mouseRelease(board.viewport(), Qt.MouseButton.MiddleButton, pos=start - QPoint(0, 70))
        self.assertGreater(board.sceneRect().bottom(), old_end)
        self.assertGreaterEqual(bar.value(), before + 69)

    def test_exclusions_round_trip_and_mask_native_pixels_in_both_crops(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "photo.png"
            Image.new("RGB", (600, 800), "gray").save(source)
            polygon = [QPointF(40, 20), QPointF(100, 20), QPointF(100, 370), QPointF(40, 370)]
            row = FaciesDetection("Слой", .8, polygon, {"__geometry_only": "1", "__geometry_id": "test"})
            exclusions = [{"radius": .018, "points": [[70 / 300, .25], [80 / 300, .25]]},
                          {"rect": [55 / 300, .4, 80 / 300, .5]}]
            record = PhotoRecord("photo", str(source), self.pixmap, [row], core_columns=self.columns,
                                 manual_exclusions=exclusions, core_columns_edited=True)
            source_bytes = source.read_bytes()
            save_project(folder / "project", "test", [record], {})
            _, records, _, missing, _ = load_project(folder / "project")
            self.assertFalse(missing)
            restored = records[0]
            self.assertEqual(restored.manual_exclusions, exclusions)
            self.assertTrue(restored.core_columns_edited)
            self.assertFalse(restored.core_columns_verified)
            crop = write_interval_crop(record, IntervalJob.create(record, row), folder / "crop.png")
            with Image.open(crop) as image:
                self.assertEqual(image.getpixel((60, 160))[3], 0)
                self.assertEqual(image.getpixel((60, 120))[3], 255)
                self.assertEqual(image.getpixel((30, 280))[3], 0)
                self.assertEqual(image.getpixel((29, 280))[3], 255)
                self.assertEqual(image.getpixel((80, 280))[3], 255)
            preview = make_interval_preview(self.pixmap, polygon, exclusions).toImage()
            self.assertEqual(preview.pixelColor(30, 80).alpha(), 0)
            self.assertEqual(preview.pixelColor(30, 60).alpha(), 255)
            self.assertEqual(preview.pixelColor(25, 160).alpha(), 0)
            self.assertEqual(preview.pixelColor(10, 160).alpha(), 255)
            self.assertEqual(source.read_bytes(), source_bytes)
