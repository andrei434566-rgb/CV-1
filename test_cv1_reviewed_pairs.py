"""A generated VLM answer becomes a target only after separate text review."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.vlm_training_data import (
    expert_description_digest,
    expert_description_is_confirmed,
    export_reviewed_pairs,
    reviewed_pair_counts,
)


class Cv1ReviewedPairsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _record(self, *, confirmed=False, training_ready=True):
        image = QImage(80, 100, QImage.Format.Format_RGB32)
        image.fill(0x665544)
        detection = FaciesDetection(
            label="Песчаник", confidence=.8,
            polygon=[QPointF(10, 10), QPointF(60, 10), QPointF(60, 80), QPointF(10, 80)],
            depth_from=100.0, depth_to=100.5, training_ready=training_ready,
            attributes={
                "__geometry_only": "1", "__geometry_id": "yolo:1",
                "Источник геометрии": "YOLO masks.xy", "Источник описания": "VLM: локальная Qwen",
                "__vlm_description": "Песчаник серый, косослоистый.",
                "__vlm_facies_association": "Предположительно русловая",
                "__vlm_depositional_environment": "не определена",
                "Литология": "Песчаник", "Текстура": "слоистая", "Цвет литологии": "#aaaaaa",
            },
        )
        if confirmed:
            self._confirm(detection)
        return PhotoRecord("one", "photo.jpg", QPixmap.fromImage(image), [detection], well_name="Р-1")

    @staticmethod
    def _confirm(detection):
        detection.attributes["__expert_description_confirmed"] = "1"
        detection.attributes["__expert_description_sha256"] = expert_description_digest(detection)

    def test_automatic_vlm_text_never_becomes_target_via_class_checkbox(self):
        record = self._record(training_ready=True)
        self.assertEqual((1, 0), reviewed_pair_counts([record]))
        self.assertFalse(expert_description_is_confirmed(record.detections[0]))

    def test_explicit_stamp_without_matching_digest_is_not_enough(self):
        record = self._record()
        item = record.detections[0]
        item.attributes["__expert_description_confirmed"] = "1"
        self.assertEqual((1, 0), reviewed_pair_counts([record]))
        item.attributes["__expert_description_sha256"] = "0" * 64
        self.assertEqual((1, 0), reviewed_pair_counts([record]))

    def test_geometry_and_text_approvals_are_both_required(self):
        record = self._record(confirmed=True, training_ready=False)
        self.assertEqual((0, 0), reviewed_pair_counts([record]))
        self.assertFalse(expert_description_is_confirmed(record.detections[0]))

    def test_whitespace_description_not_exportable_even_when_stamped(self):
        record = self._record()
        record.detections[0].attributes["__vlm_description"] = " \n "
        self._confirm(record.detections[0])
        self.assertEqual((1, 0), reviewed_pair_counts([record]))

    def test_any_meaningful_later_edit_invalidates_confirmation(self):
        original = self._record(confirmed=True).detections[0]
        changes = {
            "label": lambda item: setattr(item, "label", "Алевролит"),
            "visible_text": lambda item: item.attributes.update({"Краткое описание": "Новое описание."}),
            "vlm_text": lambda item: item.attributes.update({"__vlm_description": "Изменённый текст."}),
            "association": lambda item: item.attributes.update({"__vlm_facies_association": "Пойменная"}),
            "environment": lambda item: item.attributes.update({"__vlm_depositional_environment": "Море"}),
            "lithology": lambda item: item.attributes.update({"Литология": "Алевролит"}),
            "texture": lambda item: item.attributes.update({"Текстура": "массивная"}),
            "palette": lambda item: item.attributes.update({"Цвет литологии": "#bbbbbb"}),
            "geometry": lambda item: item.polygon[0].setX(11),
            "depth": lambda item: setattr(item, "depth_to", 101.0),
        }
        self.assertTrue(expert_description_is_confirmed(original))
        for name, change in changes.items():
            with self.subTest(change=name):
                item = copy.deepcopy(original)
                change(item)
                self.assertFalse(expert_description_is_confirmed(item))

    def test_digest_ignores_confirmation_stamps_and_internal_ids(self):
        item = self._record().detections[0]
        expected = expert_description_digest(item)
        item.attributes.update({"__expert_description_confirmed": "1", "__expert_description_sha256": expected, "__geometry_id": "new-local-id"})
        self.assertEqual(expected, expert_description_digest(item))

    def test_digest_rejects_nonfinite_geometry(self):
        item = self._record(confirmed=True).detections[0]
        item.polygon[0].setX(float("nan"))
        self.assertFalse(expert_description_is_confirmed(item))

    def test_mixed_export_audits_excel_and_expert_sources_separately(self):
        cv1 = self._record(confirmed=True)
        excel = self._record()
        item = excel.detections[0]
        item.attributes = {"Краткое описание": "Алевролит серый.", "Источник описания": "Excel: Лист1!2"}
        with tempfile.TemporaryDirectory() as directory:
            result = export_reviewed_pairs([cv1, excel], Path(directory) / "pairs")
            rows = [json.loads(line) for line in result["manifest"].read_text(encoding="utf-8").splitlines()]
            report = json.loads((result["manifest"].parent / "export_report.json").read_text(encoding="utf-8"))
            self.assertEqual(2, result["count"])
            expected = {"expert_cv1_reviewed_mask": 1, "source_direct_excel_reviewed_mask": 1}
            self.assertEqual(expected, result["pairs_by_provenance"])
            self.assertEqual(expected, report["pairs_by_provenance"])
            self.assertEqual("kern-vlm-reviewed-image-text-v2", report["schema"])
            self.assertEqual("Песчаник серый, косослоистый.", rows[0]["conversations"][1]["value"])
            self.assertEqual("expert_cv1_reviewed_mask", rows[0]["metadata"]["label_provenance"])
            self.assertEqual("source_direct_excel_reviewed_mask", rows[1]["metadata"]["label_provenance"])
            self.assertEqual(cv1.detections[0].attributes["__expert_description_sha256"], rows[0]["metadata"]["expert_description_sha256"])
            self.assertEqual("YOLO masks.xy", rows[0]["metadata"]["source_geometry"])
            self.assertEqual("Предположительно русловая", rows[0]["metadata"]["expert_reviewed_fields"]["__vlm_facies_association"])
            self.assertNotIn("__expert_description_confirmed", rows[0]["metadata"]["expert_reviewed_fields"])
            self.assertTrue(all((result["manifest"].parent / row["image"]).is_file() for row in rows))

    def test_changed_description_skipped_with_auditable_reason(self):
        accepted = self._record(confirmed=True)
        stale = self._record(confirmed=True)
        stale.detections[0].attributes["__vlm_description"] = "Автоматически заменённый текст."
        self.assertEqual((2, 1), reviewed_pair_counts([accepted, stale]))
        with tempfile.TemporaryDirectory() as directory:
            result = export_reviewed_pairs([accepted, stale], Path(directory) / "pairs")
            report = json.loads((result["manifest"].parent / "export_report.json").read_text(encoding="utf-8"))
            self.assertEqual(1, result["count"])
            self.assertEqual(1, result["skipped"])
            self.assertIn("изменилось после подтверждения", report["skipped"][0]["reason"])

    def test_cv1_marker_cannot_bypass_text_confirmation_using_excel_source(self):
        record = self._record()
        record.detections[0].attributes["Источник описания"] = "Excel: old inherited value"
        self.assertEqual((1, 0), reviewed_pair_counts([record]))

    def test_legacy_catalogue_or_vlm_text_still_not_accepted(self):
        record = self._record(confirmed=True)
        del record.detections[0].attributes["__geometry_only"]
        record.detections[0].attributes["Краткое описание"] = "Описание из справочника."
        self.assertEqual((1, 0), reviewed_pair_counts([record]))


if __name__ == "__main__":
    unittest.main()
