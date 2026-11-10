import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

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

    def test_board_drags_only_the_picked_edge(self):
        board = WorkspaceCanvas()
        self.addCleanup(board.close)
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

    def test_large_preview_resizes_and_brush_is_additive_and_undoable(self):
        dialog = PhotoQualityDialog(self.pixmap, {"label_boxes": [[.2, .2, .25, .25]]},
                                    columns=self.columns, exclusions=[])
        self.addCleanup(dialog.close)
        dialog.show()
        self.app.processEvents()
        self.drag(dialog.view, QPointF(70, 20), QPointF(70, 42))
        self.assertAlmostEqual(dialog.edited_columns()[0]["top"], 42, delta=1)
        dialog.view.set_brush(True)
        self.drag(dialog.view, QPointF(65, 80), QPointF(90, 90))
        self.assertEqual(len(dialog.boxes), 1)
        self.assertEqual(len(dialog.view.exclusions), 1)
        self.assertGreaterEqual(len(dialog.view.exclusions[0]["points"]), 2)
        self.assertAlmostEqual(dialog.edited_columns()[0]["top"], 42, delta=1)
        dialog.view.undo_stroke()
        self.assertEqual(dialog.view.exclusions, [])
        self.assertTrue(dialog.boxes[0].isVisible())

    def test_exclusions_round_trip_and_mask_native_pixels_in_both_crops(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "photo.png"
            Image.new("RGB", (600, 800), "gray").save(source)
            polygon = [QPointF(40, 20), QPointF(100, 20), QPointF(100, 370), QPointF(40, 370)]
            row = FaciesDetection("Слой", .8, polygon, {"__geometry_only": "1", "__geometry_id": "test"})
            strokes = [{"radius": .018, "points": [[70 / 300, .25], [80 / 300, .25]]}]
            record = PhotoRecord("photo", str(source), self.pixmap, [row], core_columns=self.columns,
                                 manual_exclusions=strokes, core_columns_edited=True)
            source_bytes = source.read_bytes()
            save_project(folder / "project", "test", [record], {})
            _, records, _, missing, _ = load_project(folder / "project")
            self.assertFalse(missing)
            restored = records[0]
            self.assertEqual(restored.manual_exclusions, strokes)
            self.assertTrue(restored.core_columns_edited)
            self.assertFalse(restored.core_columns_verified)
            crop = write_interval_crop(record, IntervalJob.create(record, row), folder / "crop.png")
            with Image.open(crop) as image:
                self.assertEqual(image.getpixel((60, 160))[3], 0)
                self.assertEqual(image.getpixel((60, 120))[3], 255)
            preview = make_interval_preview(self.pixmap, polygon, strokes).toImage()
            self.assertEqual(preview.pixelColor(30, 80).alpha(), 0)
            self.assertEqual(preview.pixelColor(30, 60).alpha(), 255)
            self.assertEqual(source.read_bytes(), source_bytes)
