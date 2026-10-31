"""Offline CV-1 boundary tests: real pixels, atomic fusion and preserved masks."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.cv1_analysis import (
    IntervalJob, PIPELINE_MODE, apply_interval_description, geometry_signature,
    mark_interval_failure, write_interval_crop,
)
from app.infrastructure.facies_postprocess import has_complete_core_column_coverage, postprocess_detections
from app.infrastructure.project_storage import load_project, save_project


def coordinates(detection):
    return [(point.x(), point.y()) for point in detection.polygon]


class Cv1AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        root = Path(__file__).resolve().parent / "open_datasets/test_tmp"
        root.mkdir(parents=True, exist_ok=True)
        context = tempfile.TemporaryDirectory(prefix="cv1_contract_", dir=root)
        self.addCleanup(context.cleanup)
        self.folder = Path(context.name)
        # A coordinate-dependent, non-white native image makes resampling and
        # mask/background contamination observable without a neural network.
        ys, xs = np.indices((80, 120))
        self.source = np.stack((xs + 7, ys + 11, (xs + ys) % 200 + 13), axis=-1).astype(np.uint8)
        self.photo = self.folder / "original.png"
        ok, encoded = cv2.imencode(".png", self.source)
        self.assertTrue(ok)
        self.photo.write_bytes(encoded.tobytes())
        preview = QPixmap(str(self.photo)).scaled(60, 40)
        self.detection = FaciesDetection(
            label="Не определена", confidence=0.83,
            polygon=[QPointF(5, 5), QPointF(25, 5), QPointF(5, 25)],
            attributes={"__geometry_only": "1", "__geometry_id": "yolo:1",
                        "__source_model_class": "foreign-class-name", "__interpretation_status": "waiting"},
            depth_from=1500.0, depth_to=1500.4,
        )
        self.record = PhotoRecord(
            identifier="photo-1", path=str(self.photo), pixmap=preview,
            detections=[self.detection], well_name="Р-41", analysis_mode=PIPELINE_MODE,
            core_columns=[{"left": 0, "top": 0, "right": 60, "bottom": 40}],
        )
        self.job = IntervalJob.create(self.record, self.detection)

    def response(self, **fields):
        values = {"facies_name": "Фация не установлена", "lithology": "Песчаник, предположение",
                  "texture": "Слоистая", "facies_association": "Не установлена",
                  "depositional_environment": "Недостаточно данных",
                  "description": "Серая порода с видимыми тонкими прослоями."}
        values.update(fields)
        return {"ok": True, "fields": values, "model_info": {"model_mode": "base", "base": "local-base"}}

    def test_masked_crop_uses_original_resolution_and_excludes_pixels_outside_triangle(self):
        target = self.folder / "crop.png"
        write_interval_crop(self.record, self.job, target)
        crop = cv2.imdecode(np.frombuffer(target.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(crop.shape, (41, 41, 3))
        np.testing.assert_array_equal(crop[5, 5], self.source[15, 15])
        np.testing.assert_array_equal(crop[2, 20], self.source[12, 30])
        np.testing.assert_array_equal(crop[-1, -1], np.array([255, 255, 255]))
        self.assertEqual(coordinates(self.detection), [(5, 5), (25, 5), (5, 25)])

    def test_primary_source_regime_and_provenance_are_saved_without_changing_geometry(self):
        response = self.response(hydrodynamic_regime="Предположительно: переменный поток")
        response["primary_reference"] = {"documents": [{"name": "Методика.docx", "sha256": "test"}],
                                         "scope": ["facies_name", "hydrodynamic_regime"]}
        original = coordinates(self.detection)
        apply_interval_description(self.record, self.job, response)
        self.assertEqual(self.detection.attributes["Гидродинамический режим"], "Предположительно: переменный поток")
        self.assertIn("Методика.docx", self.detection.attributes["__primary_reference"])
        self.assertEqual(coordinates(self.detection), original)
        self.assertEqual((self.detection.depth_from, self.detection.depth_to), (1500, 1500.4))

    def test_invalid_regime_cannot_partially_overwrite_interval(self):
        before = deepcopy(self.detection.attributes)
        with self.assertRaisesRegex(ValueError, "гидродинамический"):
            apply_interval_description(self.record, self.job, self.response(hydrodynamic_regime=["not a string"]))
        self.assertEqual(self.detection.attributes, before)

    def test_fusion_changes_only_draft_semantics_and_keeps_geometry_depth_and_confidence(self):
        before = (coordinates(self.detection), self.detection.depth_from,
                  self.detection.depth_to, self.detection.confidence, self.detection.stack_order)
        apply_interval_description(self.record, self.job, self.response())
        self.assertEqual((coordinates(self.detection), self.detection.depth_from, self.detection.depth_to,
                          self.detection.confidence, self.detection.stack_order), before)
        self.assertEqual(self.detection.attributes["__source_model_class"], "foreign-class-name")
        self.assertEqual(self.detection.attributes["Название породы"], "Песчаник, предположение")
        self.assertEqual(self.detection.attributes["__vlm_description"], self.response()["fields"]["description"])
        self.assertEqual(self.detection.attributes["__geometry_sha256"], self.job.geometry_sha256)
        self.assertEqual(self.detection.attributes["__interpretation_status"], "draft")
        self.assertFalse(self.detection.training_ready)

    def test_model_supplied_geometry_cannot_change_yolo_polygon(self):
        result = self.response()
        result["fields"].update(polygon=[[0, 0], [100, 0], [100, 100]], depth_from=1, confidence=1)
        result["polygon"] = [[0, 0]]
        before = coordinates(self.detection)
        apply_interval_description(self.record, self.job, result)
        self.assertEqual(coordinates(self.detection), before)
        self.assertEqual(self.detection.depth_from, 1500.0)
        self.assertEqual(self.detection.confidence, 0.83)

    def test_invalid_facies_roles_keep_observations_and_raw_output_for_review(self):
        before = coordinates(self.detection)
        apply_interval_description(self.record, self.job, self.response(
            lithology="песчаник", facies_name="песчаник", facies_association="песчаник",
            depositional_environment="вулканическая среда"))
        attrs = self.detection.attributes
        self.assertEqual(attrs["Название породы"], "песчаник")
        self.assertEqual(attrs["Фациальная ассоциация"], "Не установлена по фото")
        self.assertEqual(json.loads(attrs["__vlm_raw_fields"])["facies_association"], "песчаник")
        self.assertTrue(json.loads(attrs["__vlm_review_notes"]))
        self.assertEqual(coordinates(self.detection), before)

    def test_incomplete_error_and_empty_responses_leave_all_fields_untouched(self):
        cases = [{"ok": False, "error": "Runtime unavailable"},
                 {**self.response(), "incomplete": True},
                 {"ok": True, "fields": None},
                 self.response(description="  ")]
        for result in cases:
            with self.subTest(result=result):
                before = deepcopy((self.detection.label, self.detection.attributes, coordinates(self.detection)))
                with self.assertRaises(ValueError):
                    apply_interval_description(self.record, self.job, result)
                self.assertEqual((self.detection.label, self.detection.attributes, coordinates(self.detection)), before)

    def test_invalid_field_type_or_oversized_text_is_rejected_atomically(self):
        for result in [self.response(texture=["wrong-type"]), self.response(description="я" * 6001),
                       self.response(facies_name="а" * 1001)]:
            with self.subTest(fields=result["fields"]):
                before = deepcopy(self.detection.attributes)
                with self.assertRaises(ValueError):
                    apply_interval_description(self.record, self.job, result)
                self.assertEqual(self.detection.attributes, before)
                self.assertEqual(self.detection.label, "Не определена")

    def test_required_structured_fields_cannot_be_silently_omitted(self):
        for key in self.response()["fields"]:
            result = self.response()
            result["fields"].pop(key)
            with self.subTest(missing_field=key):
                before = deepcopy(self.detection.attributes)
                with self.assertRaises(ValueError):
                    apply_interval_description(self.record, self.job, result)
                self.assertEqual(self.detection.attributes, before)

    def test_redrawn_and_removed_intervals_reject_stale_reply(self):
        self.detection.polygon[0] = QPointF(6, 5)
        self.assertFalse(self.job.is_current(self.record))
        with self.assertRaises(ValueError):
            apply_interval_description(self.record, self.job, self.response())
        self.detection.polygon[0] = QPointF(5, 5)
        self.record.detections.clear()
        self.assertFalse(self.job.is_current(self.record))
        with self.assertRaises(ValueError):
            write_interval_crop(self.record, self.job, self.folder / "missing.png")

    def test_expert_confirmed_and_manual_intervals_cannot_be_overwritten(self):
        for protected in ("training_ready", "manual"):
            with self.subTest(protected=protected):
                self.detection.training_ready = protected == "training_ready"
                self.detection.attributes["__manual_contour"] = "1" if protected == "manual" else "0"
                before = deepcopy(self.detection.attributes)
                self.assertFalse(self.job.is_current(self.record))
                with self.assertRaises(ValueError):
                    apply_interval_description(self.record, self.job, self.response())
                self.assertEqual(self.detection.attributes, before)

    def test_expert_text_edit_during_pending_request_invalidates_reply(self):
        self.detection.attributes["__vlm_description"] = "Исправлено специалистом до ответа модели."
        self.assertFalse(self.job.is_current(self.record))
        with self.assertRaises(ValueError):
            apply_interval_description(self.record, self.job, self.response())
        self.assertEqual(self.detection.attributes["__vlm_description"],
                         "Исправлено специалистом до ответа модели.")

    def test_changed_label_or_depth_also_protects_specialist_edits_from_error_status(self):
        for field, changed in [("label", "Метка специалиста"), ("depth_from", 1499.9), ("depth_to", 1500.9)]:
            with self.subTest(field=field):
                original = getattr(self.detection, field)
                setattr(self.detection, field, changed)
                self.assertFalse(self.job.is_current(self.record))
                before = deepcopy(self.detection.attributes)
                mark_interval_failure(self.job, "Старый запрос больше не актуален")
                self.assertEqual(self.detection.attributes, before)
                setattr(self.detection, field, original)

    def test_invalid_current_geometry_is_stale_without_raising_from_guard(self):
        self.detection.polygon = [QPointF(float("nan"), 1), QPointF(2, 3), QPointF(4, 5)]
        self.assertFalse(self.job.is_current(self.record))

    def test_new_jobs_reject_nonfinite_or_insufficient_geometry(self):
        for polygon in [[], [QPointF(1, 1), QPointF(2, 2)],
                        [QPointF(1, 1), QPointF(float("inf"), 2), QPointF(3, 3)]]:
            with self.subTest(polygon=polygon):
                self.detection.polygon = polygon
                with self.assertRaises(ValueError):
                    IntervalJob.create(self.record, self.detection)

    def test_zero_area_crop_fails_without_creating_file(self):
        self.detection.polygon = [QPointF(2, 2), QPointF(3, 3), QPointF(4, 4)]
        job = IntervalJob.create(self.record, self.detection)
        target = self.folder / "degenerate.png"
        with self.assertRaises(ValueError):
            write_interval_crop(self.record, job, target)
        self.assertFalse(target.exists())

    def test_failure_is_not_a_semantic_prediction_and_keeps_expert_state(self):
        mark_interval_failure(self.job, "Видеопамять закончилась")
        self.assertEqual(self.detection.label, "Не определена")
        self.assertEqual(self.detection.attributes["__interpretation_status"], "error")
        self.assertNotIn("__vlm_description", self.detection.attributes)
        self.detection.training_ready = True
        before = deepcopy(self.detection.attributes)
        mark_interval_failure(self.job, "Не должно заменить проверенное")
        self.assertEqual(self.detection.attributes, before)

    def test_strict_postprocessing_keeps_real_gap_and_does_not_create_facies(self):
        before = coordinates(self.detection)
        self.assertFalse(has_complete_core_column_coverage(self.record))
        result = postprocess_detections(self.record, [self.detection])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], self.detection)
        self.assertEqual(coordinates(result[0]), before)
        self.assertEqual(postprocess_detections(self.record, []), [])

    def test_saved_project_retains_mode_provenance_and_real_gap_after_preview_rescale(self):
        apply_interval_description(self.record, self.job, self.response())
        self.record.analysis_issues = [{"kind": "uncovered_core", "message": "Нужна проверка покрытия"}]
        before = coordinates(self.detection)
        folder = self.folder / "saved-project"
        save_project(folder, "CV-1", [self.record], {self.record.identifier: QPointF(5, 9)})
        title, records, positions, missing, _ = load_project(folder)
        self.assertEqual(title, "CV-1")
        self.assertFalse(missing)
        self.assertEqual(len(records), 1)
        restored = records[0]
        self.assertEqual(restored.analysis_mode, PIPELINE_MODE)
        self.assertEqual(restored.analysis_issues, self.record.analysis_issues)
        self.assertEqual(restored.detections[0].attributes["__source_model_class"], "foreign-class-name")
        self.assertEqual(restored.detections[0].attributes["__vlm_description"], self.response()["fields"]["description"])
        x_ratio, y_ratio = restored.pixmap.width() / 60, restored.pixmap.height() / 40
        self.assertEqual(coordinates(restored.detections[0]), [(x * x_ratio, y * y_ratio) for x, y in before])
        self.assertFalse(has_complete_core_column_coverage(restored))
        self.assertEqual(len(postprocess_detections(restored, restored.detections)), 1)
        self.assertEqual(positions["photo-1"], QPointF(5, 9))

    def test_older_project_without_mode_remains_legacy(self):
        folder = self.folder / "legacy-project"
        manifest = save_project(folder, "Старый", [self.record], {})
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["photos"][0].pop("analysis_mode")
        manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        _, records, _, _, _ = load_project(folder)
        self.assertEqual(records[0].analysis_mode, "legacy")


if __name__ == "__main__":
    unittest.main()
