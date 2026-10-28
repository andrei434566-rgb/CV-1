"""Manual review survives autosave and is available only to the same interval."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.cv1_analysis import IntervalJob
from app.infrastructure.expert_corrections import exact_interval_edit, expert_memory_is_confirmed
from app.infrastructure.facies_mapping import facies_mapping_context
from app.infrastructure.project_storage import load_project
from app.ui.windows.main_window import MainWindow


class ExpertFaciesUiFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        p1 = patch("app.ui.windows.main_window.QSettings", return_value=settings)
        p2 = patch("app.infrastructure.expert_corrections._root", return_value=self.root / "memory")
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        self.window = MainWindow()
        self.addCleanup(self.window.close)
        image = QPixmap(100, 200)
        image.fill()
        photo = self.root / "core.png"
        self.assertTrue(image.save(str(photo)))
        self.detection = FaciesDetection("Интервал YOLO", .8, [QPointF(10, 10), QPointF(80, 10), QPointF(80, 180)],
            {"__geometry_only": "1", "__source_model_class": "core", "__source_model_class_id": "0"})
        self.record = PhotoRecord("test-photo", str(photo), image, [self.detection], well_name="Р-31")
        self.window._records = [self.record]
        self.window._project_folder = self.root / "project"
        self.window._vlm_target = (self.record, self.detection)
        self.window._active_well_name = "Р-31"
        self.window._refresh_facies_views = Mock(side_effect=lambda _record: self.window._schedule_edit_autosave())
        self.dialog = SimpleNamespace(
            selected_depth_range=lambda: (100.0, 101.0), selected_facies=lambda: "Русловая",
            selected_attributes=lambda: {**self.detection.attributes, "__vlm_description": "Серый песчаник.",
                                         "Краткое описание": "Серый песчаник."},
            training_confirmed=SimpleNamespace(isChecked=lambda: False),
            description_confirmed=SimpleNamespace(isChecked=lambda: True),
        )

    def test_edit_is_saved_confirmed_and_visible_in_exact_vlm_context(self):
        self.window._save_lithology(self.record, self.detection, self.dialog)
        self.assertTrue(expert_memory_is_confirmed(self.detection))
        self.assertFalse(self.detection.training_ready)
        self.assertFalse(IntervalJob.create(self.record, self.detection).is_current(self.record))
        self.assertTrue(self.window._edit_autosave.isActive())
        self.assertTrue(self.window._flush_edit_autosave())
        context = self.window._vlm_context()
        self.assertIn("сохранённая специалистом", context)
        self.assertIn("Исходная метка YOLO: core", context)
        self.assertIn("Серый песчаник", context)
        self.assertIsNotNone(exact_interval_edit(self.record, self.detection,
            project=self.window._expert_project_key(), well="Р-31"))
        _, records, _, missing, _ = load_project(self.window._project_folder)
        self.assertFalse(missing)
        restored = records[0].detections[0]
        self.assertTrue(expert_memory_is_confirmed(restored))
        self.assertEqual(restored.attributes["__source_model_class"], "core")
        self.assertIn("Русловая", facies_mapping_context(records[0], restored))
        self.assertEqual(len(list((self.window._project_folder / "images").iterdir())), 1)

    def test_first_edit_creates_recoverable_draft_and_migrates_memory(self):
        self.window._project_folder = None
        self.window._projects_root = lambda: self.root / "projects"
        self.window._save_lithology(self.record, self.detection, self.dialog)
        self.assertTrue(self.window._flush_edit_autosave())
        self.assertTrue((self.window._project_folder / "project.json").is_file())
        remembered = exact_interval_edit(self.record, self.detection,
                                         project=self.window._expert_project_key(), well="Р-31")
        self.assertIsNotNone(remembered)
        self.assertIn("Серый песчаник", self.window._vlm_context())

    def test_failed_autosave_keeps_unsaved_flag_for_retry(self):
        self.window._schedule_edit_autosave()
        with patch.object(self.window, "_save_current_project", return_value=False):
            self.assertFalse(self.window._flush_edit_autosave())
            self.assertTrue(self.window._edit_unsaved)
        self.assertTrue(self.window._flush_edit_autosave())
        self.assertFalse(self.window._edit_unsaved)


if __name__ == "__main__":
    unittest.main()
