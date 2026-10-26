"""CV-1 UI orchestration checks with isolated settings and CPU-only fakes."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QProcess, QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.cv1_analysis import PIPELINE_MODE, IntervalJob
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES
from app.ui.windows.main_window import MainWindow
from app.ui.dialogs.training_center_dialog import TrainingCenterDialog


_APP = QApplication.instance() or QApplication([])
_MODULE = "app.ui.windows.main_window"


class Cv1MainFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        settings_patch = patch(_MODULE + ".QSettings", return_value=self.settings)
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.window = MainWindow()
        self.addCleanup(self.close_window)
        message_patch = patch.object(self.window.assistant_panel, "append_message")
        self.messages = message_patch.start()
        self.addCleanup(message_patch.stop)
        self.window._project_folder = None

    def close_window(self):
        # Worker objects in these tests are mocks, not running Qt threads.
        self.window._segmentation_thread = None
        self.window._training_thread = None
        self.window._column_training_thread = None
        self.window._chat_photo_thread = None
        self.window._analysis_active = False
        self.window._analysis_pending = None
        self.window.close()

    def record(self, *, name="photo.png", interpreted=False):
        pixmap = QPixmap(40, 100)
        pixmap.fill()
        polygon = [QPointF(2, 5), QPointF(22, 5), QPointF(22, 55), QPointF(2, 55)]
        detection = FaciesDetection(
            UNRECOGNIZED_FACIES, .81, polygon,
            {"__geometry_only": "1", "__interpretation_status": "waiting"},
        )
        record = PhotoRecord(
            name, str(self.root / name), pixmap,
            detections=[detection] if interpreted else [],
            core_columns=[{"left": 2., "top": 5., "right": 22., "bottom": 95.}],
            analysis_mode=PIPELINE_MODE,
        )
        return record

    def valid_fields(self):
        return {
            "facies_name": "Слоистая фация", "lithology": "Песчаник",
            "texture": "Слоистая", "facies_association": "",
            "depositional_environment": "", "description": "Серый слоистый фрагмент керна.",
        }

    def runtime_config(self, mode="base"):
        python = self.root / "python.exe"
        python.write_bytes(b"not executable; transport is mocked")
        base = self.root / "base"
        base.mkdir(exist_ok=True)
        (base / "config.json").write_text("{}", encoding="utf-8")
        return {
            "python": str(python), "base": str(base), "model_mode": mode,
            "adapter": "specialist", "theory_adapter": "", "folders": [],
        }

    def test_fresh_settings_use_base_without_autodiscovering_old_adapter(self):
        config = self.window._local_vlm_config()
        self.assertEqual(config["model_mode"], "base")
        self.assertEqual(config["adapter"], "")
        self.assertEqual(config["theory_adapter"], "")

    def test_old_adapter_path_is_retained_but_requires_explicit_mode_selection(self):
        self.settings.setValue("vlm/adapter", "old-specialist")
        config = self.window._local_vlm_config()
        self.assertEqual(config["adapter"], "old-specialist")
        self.assertEqual(config["model_mode"], "base")
        self.settings.setValue("vlm/model_mode", "lora")
        self.assertEqual(self.window._local_vlm_config()["model_mode"], "lora")

    def test_selected_yolo_is_saved_to_disk_and_survives_restart_and_new_project(self):
        model = self.root / "best.pt"
        model.write_bytes(b"checkpoint fixture; actual loading is mocked")
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=lambda _: SimpleNamespace(task="segment"))}):
            self.window.select_detection_model(filename_override=model)
        # A different QSettings instance reads the durable choice, as on restart.
        restored_settings = QSettings(self.settings.fileName(), QSettings.Format.IniFormat)
        self.assertEqual(restored_settings.value("facies/model_path"), str(model.resolve()))
        with (
            patch(_MODULE + ".QSettings", return_value=restored_settings),
            patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock(side_effect=AssertionError("startup must not load weights")))}),
        ):
            restarted = MainWindow()
        try:
            self.assertEqual(restarted._resolve_model_path(), model.resolve())
            restarted.new_project()
            self.assertEqual(restarted._resolve_model_path(), model.resolve())
        finally:
            restarted.close()

    def test_missing_remembered_yolo_is_reported_once_without_picking_another_model(self):
        missing = self.root / "missing_best.pt"
        self.settings.setValue("facies/model_path", str(missing))
        with (
            patch(_MODULE + ".AssistantPanel.append_message") as append,
            patch(_MODULE + ".QFileDialog.getOpenFileName") as chooser,
        ):
            restarted = MainWindow()
            try:
                self.assertIsNone(restarted._resolve_model_path())
                restarted._show_remembered_model_status()
                restarted.new_project()
                warnings = [call.args[1] for call in append.call_args_list if "Последняя модель границ" in str(call.args[1])]
                self.assertEqual(len(warnings), 1)
                self.assertIn(str(missing), warnings[0])
                self.assertEqual(self.settings.value("facies/model_path"), str(missing))
                chooser.assert_not_called()
            finally:
                restarted.close()

    def test_invalid_new_checkpoint_does_not_replace_remembered_yolo(self):
        good = self.root / "best.pt"
        good.write_bytes(b"good fixture")
        bad = self.root / "detect_only.pt"
        bad.write_bytes(b"bad fixture")
        self.window._selected_model_path = good
        self.settings.setValue("facies/model_path", str(good))
        with (
            patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=lambda _: SimpleNamespace(task="detect"))}),
            patch(_MODULE + ".QMessageBox.warning") as warning,
        ):
            self.window.select_detection_model(filename_override=bad)
        warning.assert_called_once()
        self.assertEqual(self.window._resolve_model_path(), good)
        self.assertEqual(self.settings.value("facies/model_path"), str(good))

    def test_import_chat_photos_reuses_existing_record_and_deduplicates_paths(self):
        existing = self.record()
        self.window._records = [existing]
        new = self.root / "second.png"
        with (
            patch(_MODULE + ".load_working_pixmap", return_value=QPixmap(40, 100)),
            patch.object(self.window.workspace, "add_photos") as add,
            patch.object(self.window, "_update_header_context"),
            patch.object(self.window, "_refresh_project_tree"),
            patch.object(self.window, "run_segmentation") as run,
        ):
            self.window._import_chat_analysis_photos([Path(existing.path), new, new])
        self.assertEqual(2, len(self.window._records))
        self.assertIs(self.window._records[0], existing)
        self.assertEqual(1, len(add.call_args.args[0]))
        self.assertEqual(2, len(run.call_args.args[0]))
        self.assertTrue(all(record.analysis_mode == PIPELINE_MODE for record in self.window._records))

    def test_existing_strict_chat_photo_does_not_launch_duplicate_segmentation(self):
        record = self.record(interpreted=True)
        self.window._records = [record]
        self.window._chat_photo_path = Path(record.path)
        with (
            patch.object(self.window, "_resolve_model_path", return_value=self.root / "best.pt"),
            patch.object(self.window, "_import_chat_analysis_photos") as import_photos,
            patch.object(self.window, "_send_vlm_request") as send,
        ):
            self.window._analyze_chat_photo("Что видно на фото?")
        import_photos.assert_not_called()
        self.assertEqual(send.call_args.args[:2], ("photo_chat", "Что видно на фото?"))

    def test_chat_retry_reuses_yolo_masks_and_only_resets_unconfirmed_errors(self):
        record = self.record(interpreted=True)
        retry = record.detections[0]
        retry.attributes["__interpretation_status"] = "error"
        confirmed = self.record(name="confirmed.png", interpreted=True).detections[0]
        confirmed.attributes["__interpretation_status"] = "error"
        confirmed.training_ready = True
        manual = self.record(name="manual.png", interpreted=True).detections[0]
        manual.attributes.update({"__interpretation_status": "error", "__manual_contour": "1"})
        record.detections += [confirmed, manual]
        original = [(point.x(), point.y()) for point in retry.polygon]
        self.window._records = [record]
        self.window._chat_photo_path = Path(record.path)
        with (
            patch.object(self.window, "_resolve_model_path", return_value=self.root / "best.pt"),
            patch.object(self.window, "_import_chat_analysis_photos") as import_photos,
            patch.object(self.window, "run_segmentation") as segment,
            patch.object(self.window, "_start_cv1_descriptions") as describe,
            patch.object(self.window, "_send_vlm_request") as send,
        ):
            self.window._analyze_chat_photo("Попробуй описать ещё раз")
        describe.assert_called_once()
        import_photos.assert_not_called()
        segment.assert_not_called()
        send.assert_not_called()
        self.assertEqual(retry.attributes["__interpretation_status"], "waiting")
        self.assertEqual(confirmed.attributes["__interpretation_status"], "error")
        self.assertEqual(manual.attributes["__interpretation_status"], "error")
        self.assertEqual([(point.x(), point.y()) for point in retry.polygon], original)

    def test_chat_with_recorded_no_masks_issue_does_not_repeat_yolo_on_each_question(self):
        record = self.record()
        record.analysis_issues = [{"kind": "no_masks", "message": "Модель не вернула масок"}]
        self.window._records = [record]
        self.window._chat_photo_path = Path(record.path)
        with (
            patch.object(self.window, "_resolve_model_path", return_value=self.root / "best.pt"),
            patch.object(self.window, "_import_chat_analysis_photos") as import_photos,
            patch.object(self.window, "run_segmentation") as segment,
            patch.object(self.window, "_start_cv1_descriptions") as describe,
            patch.object(self.window, "_send_vlm_request") as send,
        ):
            self.window._analyze_chat_photo("Что можно увидеть на фото?")
        import_photos.assert_not_called()
        segment.assert_not_called()
        describe.assert_not_called()
        self.assertEqual(send.call_args.args[:2], ("photo_chat", "Что можно увидеть на фото?"))
        self.assertEqual(record.detections, [])

    def test_chat_source_link_finds_stable_record_after_project_save_changes_path(self):
        record = self.record(interpreted=True)
        source = Path(record.path)
        self.window._chat_record_ids[str(source.resolve())] = record.identifier
        record.path = str(self.root / "saved_project" / "images" / "copied_photo.png")
        self.window._records = [record]
        self.window._chat_photo_path = source
        self.assertIs(self.window._record_for_chat_photo(source), record)
        with (
            patch.object(self.window, "_resolve_model_path", return_value=self.root / "best.pt"),
            patch.object(self.window, "_import_chat_analysis_photos") as import_photos,
            patch.object(self.window, "_send_vlm_request") as send,
        ):
            self.window._analyze_chat_photo("Опиши сохранённое фото")
        import_photos.assert_not_called()
        self.assertEqual(len(self.window._records), 1)
        self.assertEqual(send.call_args.args[:2], ("photo_chat", "Опиши сохранённое фото"))

    def test_pt_attachment_routes_to_model_loader_not_unsupported_file(self):
        model = self.root / "best.pt"
        with patch.object(self.window, "select_detection_model") as select:
            self.window._handle_chat_files([str(model)])
        select.assert_called_once_with(filename_override=model)
        self.assertFalse(any("не поддерживается" in str(call) for call in self.messages.call_args_list))

    def test_segmentation_worker_gets_geometry_only_and_stops_vlm_first(self):
        record = self.record()
        self.window._records = [record]
        events = []
        with (
            patch.object(self.window, "_resolve_model_path", return_value=self.root / "best.pt"),
            patch.object(self.window, "_resolve_core_column_model_path", return_value=None),
            patch.object(self.window._vlm_client, "stop", side_effect=lambda: events.append("stop")),
            patch.object(self.window, "_show_activity"),
            patch.object(self.window, "_refresh_training_bar"),
            patch(_MODULE + ".QThread") as thread,
            patch(_MODULE + ".SegmentationWorker", side_effect=lambda *args, **kwargs: (events.append("worker"), Mock())[1]) as worker,
        ):
            self.window.run_segmentation([record])
        self.assertEqual(events[:2], ["stop", "worker"])
        self.assertIs(worker.call_args.kwargs["geometry_only"], True)
        self.assertEqual(worker.call_args.args[1][0][0], record.path)
        thread.return_value.start.assert_called_once()

    def test_no_model_prepares_columns_without_inventing_facies_bands(self):
        record = self.record()
        self.window._records = [record]
        with (
            patch.object(self.window, "_resolve_model_path", return_value=None),
            patch.object(self.window, "_suggest_core_columns", return_value=record.core_columns),
            patch.object(self.window, "_fill_unclassified_columns") as fill,
            patch.object(self.window, "_show_stack"),
            patch.object(self.window.workspace, "update_photo_detections"),
            patch.object(self.window, "_refresh_project_tree"),
            patch.object(self.window, "_refresh_training_bar"),
        ):
            self.window.run_segmentation([record])
        fill.assert_not_called()
        self.assertEqual(record.detections, [])
        self.assertEqual(record.analysis_mode, PIPELINE_MODE)

    def test_show_stack_does_not_rewrite_strict_polygons_columns_or_labels(self):
        record = self.record(interpreted=True)
        self.window._records = [record]
        original_columns = deepcopy(record.core_columns)
        original_polygon = [(point.x(), point.y()) for point in record.detections[0].polygon]
        with (
            patch.object(self.window, "_suggest_core_columns") as columns,
            patch.object(self.window, "_fill_unclassified_columns") as fill,
            patch(_MODULE + ".postprocess_detections") as postprocess,
            patch.object(self.window, "_sync_facies_palette"),
            patch.object(self.window.workspace, "show_correlation"),
        ):
            self.window._show_stack()
        columns.assert_not_called()
        fill.assert_not_called()
        postprocess.assert_not_called()
        self.assertEqual(original_columns, record.core_columns)
        self.assertEqual(original_polygon, [(point.x(), point.y()) for point in record.detections[0].polygon])
        self.assertEqual(record.detections[0].label, UNRECOGNIZED_FACIES)

    def test_depth_edit_without_calibration_does_not_move_strict_polygon(self):
        record = self.record(interpreted=True)
        detection = record.detections[0]
        detection.depth_from, detection.depth_to = 100., 101.
        record.depth_segments = deepcopy(record.core_columns)
        original = [(point.x(), point.y()) for point in detection.polygon]
        self.assertFalse(self.window._sync_polygon_to_depth_range(record, detection))
        self.assertFalse(self.window._sync_depth_range_from_polygon(record, detection))
        self.assertEqual([(point.x(), point.y()) for point in detection.polygon], original)
        self.assertEqual((detection.depth_from, detection.depth_to), (100., 101.))

    def test_reopening_strict_project_does_not_add_catalogue_interpretation(self):
        record = self.record(interpreted=True)
        record.detections[0].label = "Dch"
        record.detections[0].attributes["__vlm_description"] = "Описание только видимых признаков."
        expected = deepcopy(record.detections[0].attributes)
        with (
            patch(_MODULE + ".load_project", return_value=("Сохранённый проект", [record], {}, [], [])),
            patch(_MODULE + ".load_catalog_snapshot", return_value=None),
            patch.object(self.window, "_load_global_primary_catalog", return_value=None),
            patch.object(self.window, "_resolve_training_class", return_value={"metadata": {"Интерпретация": "Шаблон справочника"}}) as resolve,
            patch.object(self.window.workspace, "restore_photo"),
            patch.object(self.window, "_show_stack"),
            patch.object(self.window, "_refresh_project_tree"),
            patch.object(self.window, "_refresh_training_bar"),
        ):
            self.window._open_project(self.root)
        resolve.assert_not_called()
        self.assertEqual(record.detections[0].attributes, expected)

    def test_interval_request_passes_mode_and_uses_crop_without_local_rag(self):
        record = self.record(interpreted=True)
        self.window._records = [record]
        self.window._analysis_active = True
        self.window._analysis_total = 1
        job = IntervalJob.create(record, record.detections[0])
        self.window._analysis_queue = deque([job])
        with (
            patch(_MODULE + ".user_data_root", return_value=self.root),
            patch(_MODULE + ".write_interval_crop", side_effect=lambda _record, _job, path: path.parent.mkdir(parents=True, exist_ok=True) or path.write_bytes(b"crop")),
            patch.object(self.window, "_local_vlm_config", return_value=self.runtime_config("lora")),
            patch.object(self.window._vlm_client, "send", return_value="request-one") as send,
            patch.object(self.window, "_update_activity"),
        ):
            self.window._advance_cv1_description()
        payload = send.call_args.args[1]
        self.assertEqual(payload["operation"], "describe")
        self.assertEqual(payload["model_mode"], "lora")
        self.assertEqual(payload["adapter"], "specialist")
        self.assertFalse(payload["use_knowledge"])
        self.assertEqual(payload["folders"], [])
        self.assertNotEqual(payload["image"], record.path)
        self.assertEqual(self.window._analysis_pending[:2], ("request-one", job))
        self.assertIsNone(self.window._vlm_pending)

    def test_regular_vlm_request_also_carries_explicit_mode(self):
        with (
            patch.object(self.window, "_local_vlm_config", return_value=self.runtime_config("base")),
            patch.object(self.window, "_knowledge_folders", return_value=[]),
            patch(_MODULE + ".related_text_corrections", return_value=[]),
            patch.object(self.window._vlm_client, "send", return_value="ordinary") as send,
        ):
            self.window._send_vlm_request("chat", "Что такое аркоз?")
        self.assertEqual(send.call_args.args[1]["model_mode"], "base")
        self.assertEqual(self.window._vlm_pending[0], "ordinary")

    def test_description_stage_starts_after_segmentation_thread_is_cleared(self):
        record = self.record(interpreted=True)
        self.window._records = [record]
        self.window._segmentation_thread = Mock()
        with (
            patch.object(self.window, "_refresh_project_tree"),
            patch.object(self.window, "_refresh_training_bar"),
            patch.object(self.window, "_close_activity"),
            patch.object(self.window, "_show_stack"),
            patch.object(self.window, "_start_cv1_descriptions", side_effect=lambda: self.assertIsNone(self.window._segmentation_thread)) as start,
        ):
            self.window._on_segmentation_finished()
        start.assert_called_once()

    def test_failed_segmentation_does_not_start_description_stage(self):
        self.window._segmentation_failed = True
        with (
            patch.object(self.window, "_refresh_project_tree"),
            patch.object(self.window, "_refresh_training_bar"),
            patch.object(self.window, "_close_activity"),
            patch.object(self.window, "_start_cv1_descriptions") as start,
        ):
            self.window._on_segmentation_finished()
        start.assert_not_called()

    def test_only_waiting_unconfirmed_strict_intervals_enter_description_queue(self):
        waiting = self.record(name="waiting.png", interpreted=True)
        reviewed = self.record(name="reviewed.png", interpreted=True)
        reviewed.detections[0].training_ready = True
        draft = self.record(name="draft.png", interpreted=True)
        draft.detections[0].attributes["__interpretation_status"] = "draft"
        legacy = self.record(name="legacy.png", interpreted=True)
        legacy.analysis_mode = "legacy"
        self.window._records = [waiting, reviewed, draft, legacy]
        with patch(_MODULE + ".QTimer.singleShot"), patch.object(self.window, "_show_activity"):
            self.window._start_cv1_descriptions()
        self.assertEqual(self.window._analysis_total, 1)
        self.assertEqual([job.record_id for job in self.window._analysis_queue], [waiting.identifier])

    def test_deleted_queued_interval_is_skipped_without_request(self):
        record = self.record(interpreted=True)
        job = IntervalJob.create(record, record.detections[0])
        self.window._records = [record]
        record.detections.clear()
        self.window._analysis_active = True
        self.window._analysis_queue = deque([job])
        with (
            patch(_MODULE + ".user_data_root", return_value=self.root),
            patch(_MODULE + ".write_interval_crop") as crop,
            patch.object(self.window._vlm_client, "send") as send,
            patch.object(self.window, "_finish_cv1_descriptions") as finish,
        ):
            self.window._advance_cv1_description()
        crop.assert_not_called()
        send.assert_not_called()
        finish.assert_called_once()
        self.assertEqual(self.window._analysis_done, 1)
        self.assertEqual(self.window._analysis_success, 0)

    def test_interval_reply_is_correlated_and_applied_exactly_once(self):
        record = self.record(interpreted=True)
        self.window._records = [record]
        self.window._analysis_active = True
        self.window._analysis_total = 1
        job = IntervalJob.create(record, record.detections[0])
        crop = self.root / "crop.png"
        crop.write_bytes(b"temporary")
        self.window._analysis_pending = ("interval-id", job, crop)
        result = {"id": "interval-id", "ok": True, "fields": self.valid_fields()}
        with (
            patch(_MODULE + ".QTimer.singleShot"),
            patch.object(self.window.workspace, "update_photo_detections"),
        ):
            self.window._on_vlm_reply({**result, "id": "unrelated-id"})
            self.assertIsNotNone(self.window._analysis_pending)
            self.assertTrue(crop.is_file())
            self.assertEqual(self.window._analysis_done, 0)
            self.window._on_vlm_reply(result)
            self.window._on_vlm_reply(result)
        self.assertEqual(self.window._analysis_done, 1)
        self.assertEqual(self.window._analysis_success, 1)
        self.assertFalse(crop.exists())
        self.assertEqual(record.detections[0].attributes["__vlm_description"], self.valid_fields()["description"])
        self.assertFalse(record.detections[0].training_ready)

    def test_reply_cannot_overwrite_specialist_confirmation_made_while_waiting(self):
        record = self.record(interpreted=True)
        detection = record.detections[0]
        self.window._records = [record]
        job = IntervalJob.create(record, detection)
        crop = self.root / "crop.png"
        crop.write_bytes(b"temporary")
        self.window._analysis_pending = ("interval-id", job, crop)
        detection.label = "Подтверждённая фация"
        detection.training_ready = True
        detection.attributes["__vlm_description"] = "Исправление специалиста."
        expected = deepcopy(detection.attributes)
        with patch(_MODULE + ".QTimer.singleShot"), patch.object(self.window.workspace, "update_photo_detections"):
            self.window._on_vlm_reply({"id": "interval-id", "ok": True, "fields": self.valid_fields()})
        self.assertEqual(detection.label, "Подтверждённая фация")
        self.assertEqual(detection.attributes, expected)
        self.assertTrue(detection.training_ready)
        self.assertEqual(self.window._analysis_success, 0)

    def test_success_and_failed_intervals_are_reported_separately(self):
        self.window._analysis_active = True
        self.window._analysis_total = 3
        self.window._analysis_success = 1
        self.window._analysis_done = 3
        self.window._analysis_geometry_issues = [{"photo": "photo.png", "kind": "gap", "message": "Есть незакрытый интервал"}]
        with (
            patch.object(self.window, "_show_stack"),
            patch.object(self.window, "_refresh_project_tree"),
            patch.object(self.window, "_refresh_training_bar"),
            patch.object(self.window, "_close_activity"),
        ):
            self.window._finish_cv1_descriptions()
        text = self.messages.call_args.args[1]
        self.assertIn("1 из 3", text)
        self.assertIn("черновики", text)
        self.assertIn("Пропуски", text)
        self.assertFalse(self.window._analysis_active)

    def test_missing_runtime_stops_queue_and_marks_errors_without_fake_text(self):
        first = self.record(name="one.png", interpreted=True)
        second = self.record(name="two.png", interpreted=True)
        self.window._records = [first, second]
        self.window._analysis_queue = deque(IntervalJob.create(record, record.detections[0]) for record in self.window._records)
        self.window._analysis_active = True
        self.window._analysis_total = 2
        config = {"python": str(self.root / "missing.exe"), "base": str(self.root / "missing"), "adapter": "", "theory_adapter": "", "model_mode": "base"}
        with (
            patch(_MODULE + ".user_data_root", return_value=self.root),
            patch(_MODULE + ".write_interval_crop"),
            patch.object(self.window, "_local_vlm_config", return_value=config),
            patch.object(self.window._vlm_client, "send") as send,
            patch.object(self.window, "_finish_cv1_descriptions") as finish,
        ):
            self.window._advance_cv1_description()
        send.assert_not_called()
        finish.assert_called_once()
        self.assertEqual(self.window._analysis_done, 2)
        self.assertEqual(self.window._analysis_success, 0)
        self.assertFalse(self.window._analysis_queue)
        for record in (first, second):
            self.assertEqual(record.detections[0].attributes["__interpretation_status"], "error")
            self.assertNotIn("__vlm_description", record.detections[0].attributes)

    def test_training_center_cannot_start_qwen_during_analysis_or_segmentation(self):
        for analysis, segmentation in ((True, None), (False, Mock())):
            with self.subTest(analysis=analysis):
                main = SimpleNamespace(_analysis_active=analysis, _segmentation_thread=segmentation)
                process = Mock()
                process.state.return_value = QProcess.ProcessState.NotRunning
                dialog = SimpleNamespace(main=main, process=process)
                with patch("app.ui.dialogs.training_center_dialog.QMessageBox.information") as message:
                    TrainingCenterDialog._train(dialog, "pairs")
                message.assert_called_once()
                process.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
