"""CV-1 geometry route never substitutes class priors or invented masks."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtGui import QPolygonF

from app.infrastructure.ml.yolo_model_service import SegmentationWorker, YoloModelService


def _result(polygons, *, names=None, masks=True):
    boxes = [
        SimpleNamespace(cls=np.array(index), conf=np.array(.9), xyxy=np.array([[0, 0, 100, 100]]))
        for index in range(len(polygons))
    ]
    return SimpleNamespace(
        orig_shape=(100, 100), boxes=boxes,
        names=names or {index: "DWCh@80" for index in range(len(polygons))},
        masks=SimpleNamespace(xy=[np.asarray(points, dtype=float) for points in polygons]) if masks else None,
    )


class YoloGeometryContractTests(unittest.TestCase):
    def _service(self, results, *, task="segment"):
        fake_yolo = Mock(return_value=results)
        fake_yolo.task = task
        fake_yolo.ckpt = {"core_description_checkpoint": {"must_not_load": True}}
        module = types.ModuleType("ultralytics")
        module.YOLO = Mock(return_value=fake_yolo)
        with (
            patch.dict(sys.modules, {"ultralytics": module}),
            patch.object(YoloModelService, "_best_device", return_value=("cpu", "CPU")),
            patch.object(YoloModelService, "_load_facies_catalog", side_effect=AssertionError("catalogue must not enter geometry")),
        ):
            service = YoloModelService(Path("best.pt"), geometry_only=True)
        return service

    def test_ctor_skips_embedded_description_and_catalogue(self):
        service = self._service([])
        self.assertIsNone(service._description_generator)
        self.assertEqual({}, service._facies_catalog)

    def test_detection_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "task=segment"):
            self._service([], task="detect")

    def test_real_masks_required_even_if_model_task_unknown(self):
        service = self._service([_result([[(1, 1), (10, 1), (10, 10)]], masks=False)], task="")
        with self.assertRaisesRegex(ValueError, "masks.xy"):
            service.predict("unused.png", (100, 100), [])

    def test_missing_mask_for_one_box_is_error_not_bbox_fallback(self):
        result = _result([[(1, 1), (10, 1), (10, 10)]])
        result.masks.xy.clear()
        service = self._service([result])
        with self.assertRaisesRegex(ValueError, "masks.xy"):
            service.predict("unused.png", (100, 100), [])

    def test_real_triangle_scaled_without_semantic_leaks_or_gap_fill(self):
        service = self._service([_result([[(10, 10), (40, 10), (10, 40)]])])
        columns = [{"left": 0, "top": 0, "right": 100, "bottom": 100}]
        with (
            patch.object(service, "_predict_structure_intervals", side_effect=AssertionError("no heuristic intervals")),
            patch.object(service, "_attach_generated_descriptions", side_effect=AssertionError("no text head")),
            patch.object(service, "_facies_attributes", side_effect=AssertionError("no facies meaning")),
            patch("app.infrastructure.ml.yolo_model_service.complete_core_column_coverage", side_effect=AssertionError("no filled gaps")),
        ):
            output = service.predict("unused.png", (200, 200), columns)
        self.assertEqual(1, len(output))
        item = output[0]
        self.assertEqual("Интервал YOLO", item.label)
        self.assertEqual("1", item.attributes["__geometry_only"])
        self.assertEqual("DWCh@80", item.attributes["__source_model_class"])
        self.assertEqual("ожидает VLM", item.attributes["Статус интерпретации"])
        self.assertFalse(item.training_ready)
        self.assertEqual({}, item.alternatives)
        self.assertNotIn("Код фации", item.attributes)
        self.assertNotIn("Краткое описание", item.attributes)
        self.assertNotIn("Литология", item.attributes)
        self.assertAlmostEqual(1800, service._path_area(service._polygon_path(item.polygon)))
        self.assertTrue(any(issue["kind"] == "uncovered_core" for issue in service.geometry_issues))

    def test_mask_spanning_two_columns_is_split_without_bridge(self):
        service = self._service([_result([[(10, 10), (90, 10), (90, 60), (10, 60)]])])
        columns = [
            {"left": 10, "top": 0, "right": 30, "bottom": 100},
            {"left": 70, "top": 0, "right": 90, "bottom": 100},
        ]
        output = service.predict("unused.png", (100, 100), columns)
        self.assertEqual(2, len(output))
        self.assertEqual({"0", "1"}, {item.attributes["__core_column_index"] for item in output})
        self.assertEqual(2, len({item.attributes["__geometry_id"] for item in output}))
        self.assertTrue(all(QPolygonF(item.polygon).boundingRect().width() <= 20 for item in output))
        self.assertEqual(2000, sum(service._path_area(service._polygon_path(item.polygon)) for item in output))

    def test_overlap_reported_without_erasing_either_real_mask(self):
        service = self._service([_result([
            [(10, 10), (40, 10), (40, 50), (10, 50)],
            [(20, 20), (50, 20), (50, 60), (20, 60)],
        ])])
        output = service.predict("unused.png", (100, 100), [])
        self.assertEqual(2, len(output))
        self.assertTrue(any(issue["kind"] == "overlapping_masks" for issue in service.geometry_issues))
        self.assertTrue(all(not item.alternatives for item in output))

    def test_empty_prediction_reports_uncovered_core_without_fake_facies(self):
        service = self._service([_result([])])
        columns = [{"left": 10, "top": 0, "right": 30, "bottom": 100}]
        self.assertEqual([], service.predict("unused.png", (100, 100), columns))
        self.assertEqual({"uncovered_core", "no_masks"}, {issue["kind"] for issue in service.geometry_issues})
        self.assertEqual(2000, service.geometry_issues[0]["area_px2"])

    def test_complete_real_coverage_has_no_missing_region(self):
        service = self._service([_result([[(10, 0), (30, 0), (30, 100), (10, 100)]])])
        output = service.predict("unused.png", (100, 100), [{"left": 10, "top": 0, "right": 30, "bottom": 100}])
        self.assertEqual(1, len(output))
        self.assertEqual([], service.geometry_issues)

    def test_invalid_nan_mask_is_not_a_reportable_geology_interval(self):
        service = self._service([_result([[(10, 10), (float("nan"), 10), (10, 40)]])])
        self.assertEqual([], service.predict("unused.png", (100, 100), []))
        self.assertTrue(any(issue["kind"] == "invalid_mask" for issue in service.geometry_issues))

    def test_background_removed_without_filling_its_area_with_a_class(self):
        service = self._service([_result([[(10, 10), (30, 10), (30, 40), (10, 40)]], names={0: "shlak"})])
        self.assertEqual([], service.predict("unused.png", (100, 100), [{"left": 10, "top": 0, "right": 30, "bottom": 100}]))
        self.assertTrue(any(issue["kind"] == "uncovered_core" for issue in service.geometry_issues))

    def test_worker_forwards_geometry_issues_then_releases_cuda_before_finished(self):
        columns = [{"left": 10, "top": 0, "right": 30, "bottom": 100}]
        fake_service = SimpleNamespace(
            device_label="CPU", image_size=640, confidence_threshold=.5,
            predict=Mock(return_value=[]),
            geometry_issues=[{"kind": "no_masks", "message": "no polygons"}],
        )
        cuda = SimpleNamespace(is_available=lambda: True, empty_cache=Mock())
        torch = types.ModuleType("torch")
        torch.cuda = cuda
        worker = SegmentationWorker(Path("best.pt"), [("photo.png", 100, 100, columns, True)], geometry_only=True)
        events = []
        worker.geometry_issues_ready.connect(lambda photo, issues: events.append((photo, issues)))
        worker.finished.connect(lambda: events.append(("finished", cuda.empty_cache.call_count)))
        with (
            patch("app.infrastructure.ml.yolo_model_service.YoloModelService", return_value=fake_service) as factory,
            patch("app.infrastructure.ml.yolo_model_service.CoreColumnRecognizer"),
            patch.dict(sys.modules, {"torch": torch}),
        ):
            worker.run()
        self.assertTrue(factory.call_args.kwargs["geometry_only"])
        self.assertEqual(("photo.png", fake_service.geometry_issues), events[0])
        self.assertEqual(("finished", 1), events[-1])
        fake_service.predict.assert_called_once_with("photo.png", (100, 100), columns)


if __name__ == "__main__":
    unittest.main()
