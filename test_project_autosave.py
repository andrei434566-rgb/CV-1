"""Project saves must preserve the last manifest and avoid photo duplication."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.project_storage import load_project, save_project


class ProjectAutosaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        picture = QPixmap(32, 48)
        picture.fill()
        self.source = self.root / "original.png"
        self.assertTrue(picture.save(str(self.source)))
        self.detection = FaciesDetection("A", 0.8, [QPointF(1, 1), QPointF(20, 1), QPointF(20, 30)])
        self.record = PhotoRecord("one", str(self.source), picture, [self.detection], well_name="Р-31")
        self.folder = self.root / "project"

    def test_repeated_save_keeps_one_photo_and_restores_last_edit(self):
        save_project(self.folder, "Тест", [self.record], {})
        saved_path = self.record.path
        self.detection.label = "Исправленная фация"
        self.detection.attributes["Краткое описание"] = "Серый песчаник"
        save_project(self.folder, "Тест", [self.record], {})
        save_project(self.folder, "Тест", [self.record], {})
        self.assertEqual(self.record.path, saved_path)
        self.assertEqual(len(list((self.folder / "images").iterdir())), 1)
        _, records, _, missing, _ = load_project(self.folder)
        self.assertFalse(missing)
        self.assertEqual(records[0].detections[0].label, "Исправленная фация")
        self.assertEqual(records[0].detections[0].attributes["Краткое описание"], "Серый песчаник")

    def test_failed_manifest_replace_keeps_prior_valid_project(self):
        manifest = save_project(self.folder, "Тест", [self.record], {})
        old = manifest.read_bytes()
        self.detection.label = "Новое название"
        with patch.object(Path, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                save_project(self.folder, "Тест", [self.record], {})
        self.assertEqual(manifest.read_bytes(), old)
        self.assertEqual(list(self.folder.glob(".project.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
