"""Small offline checks for the reviewed-pair and theory split."""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from openpyxl import Workbook
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.vlm_training_data import export_reviewed_pairs, export_theory_qa, reviewed_pair_counts


class VlmTrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _record(self, approved: bool):
        image = QImage(80, 100, QImage.Format.Format_RGB32)
        image.fill(0x665544)
        detection = FaciesDetection(
            label="Песчаник", confidence=1.0,
            polygon=[QPointF(10, 10), QPointF(60, 10), QPointF(60, 80), QPointF(10, 80)],
            training_ready=approved,
            attributes={"Краткое описание": "Песчаник серый, косослоистый.", "Источник описания": "Excel: Лист1, строка 2"},
        )
        return PhotoRecord("id", "photo.jpg", QPixmap.fromImage(image), [detection], well_name="Р-1")

    def test_export_only_approved_photo_excel_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            unreviewed = self._record(False)
            approved = self._record(True)
            self.assertEqual(reviewed_pair_counts([unreviewed, approved]), (1, 1))
            result = export_reviewed_pairs([unreviewed, approved], Path(temporary) / "pairs")
            self.assertEqual(result["count"], 1)
            row = json.loads(result["manifest"].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["conversations"][1]["value"], "Песчаник серый, косослоистый.")
            self.assertTrue((result["manifest"].parent / row["image"]).is_file())
            self.assertEqual(row["metadata"]["source_row"], "Excel: Лист1, строка 2")

    def test_theory_qa_is_text_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Вопрос", "Ответ"])
            sheet.append(["Что такое градационная слоистость?", "Изменение крупности зёрен по мощности слоя."])
            workbook.save(root / "qa.xlsx")
            result = export_theory_qa(root / "qa.xlsx", root / "theory.jsonl")
            self.assertEqual(result["count"], 1)
            row = json.loads(result["manifest"].read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("image", row)
            self.assertEqual(row["metadata"]["record_type"], "text_only")

    def test_unreviewed_not_exported(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                export_reviewed_pairs([self._record(False)], Path(temporary) / "pairs")


if __name__ == "__main__":
    unittest.main()
