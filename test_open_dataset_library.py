"""Library readiness, relocation and explicit chat routing regressions."""

from __future__ import annotations

import json
import builtins
import importlib.util
import os
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.infrastructure.local_chat_intents import wants_open_datasets
from app.infrastructure.open_dataset_library import last_preparation, prepare_source, prepared_knowledge_folders, read_state, register_files, write_state
from app.infrastructure.ml import open_dataset_train
from app.infrastructure.ml.open_dataset_train import prepared_training_input

_TEMP_ROOT = Path(__file__).resolve().parent / "open_datasets" / "test_tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class LibraryTests(unittest.TestCase):
    def test_new_input_invalidates_previous_preparation(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            first, second = root / "first.zip", root / "second.zip"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            state = register_files(root, "unibo", [first])
            state["preparation"] = {"output_dir": str(root / "prepared")}
            write_state(root, "unibo", state)
            self.assertIn("preparation", register_files(root, "unibo", [first]))
            self.assertNotIn("preparation", register_files(root, "unibo", [second]))

    def test_relocated_preparation_and_missing_manifest(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            actual = root / "unibo" / "prepared" / "run1"
            actual.mkdir(parents=True)
            (actual / "summary.json").write_text("{}")
            (actual / "manifest.jsonl").write_text("")
            old = root / "old_workspace" / "run1"
            write_state(root, "unibo", {"preparation": {"output_dir": str(old), "manifest": str(old / "manifest.jsonl")}})
            self.assertEqual(str(actual), last_preparation(root, "unibo")["output_dir"])
            (actual / "manifest.jsonl").unlink()
            self.assertIsNone(last_preparation(root, "unibo"))

    def test_interval_string_progress_is_normalized(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            source = root / "source.zip"
            source.write_bytes(b"source")
            events = []
            def importer(source_id, paths, destination, *, progress, cancel):
                destination.mkdir(parents=True)
                (destination / "report.json").write_text("{}")
                manifest = destination / "candidates.jsonl"
                manifest.write_text("")
                progress("Исходный документ")
                return {"manifest": str(manifest), "readiness": {"can_train": False}}
            with patch("app.infrastructure.open_dataset_intervals.prepare_interval_dataset", side_effect=importer):
                prepare_source(root, "ukcs", [source], progress=events.append)
            self.assertEqual([{"message": "Исходный документ"}], events)
            self.assertIsNotNone(last_preparation(root, "ukcs"))

    def test_training_does_not_follow_old_absolute_paths(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            for split in ("train", "val"):
                directory = root / "images" / split
                directory.mkdir(parents=True)
                (directory / "example.jpg").write_bytes(b"image")
            (root / "data.yaml").write_text("names: [core_column]\n")
            (root / "summary.json").write_text(json.dumps({
                "task": "segmentation", "readiness": "ready", "data_yaml": "Z:/old/data.yaml",
            }))
            self.assertEqual(str(root / "data.yaml"), prepared_training_input(root)["data"])
            summary = json.loads((root / "summary.json").read_text())
            summary["readiness"] = "needs_validation"
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                prepared_training_input(root)

    def test_moved_multi_root_classification_and_changed_input(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            actual = root / "dcid" / "prepared" / "run1"
            old = root / "missing" / "run1"
            old.mkdir(parents=True)  # A copy must not keep reading another workspace.
            for split in ("train", "val"):
                folder = actual / "classification" / "DCID-7" / split / "sandstone"
                folder.mkdir(parents=True)
                (folder / "example.jpg").write_bytes(b"image")
            summary = {"output_dir": str(old), "task": "classification", "readiness": "ready",
                       "classification_roots": [str(old / "classification" / "DCID-7")]}
            (actual / "summary.json").write_text(json.dumps(summary))
            (actual / "manifest.jsonl").write_text("")
            source = root / "source.zip"
            source.write_bytes(b"original")
            state = register_files(root, "dcid", [source])
            state["preparation"] = {**summary, "manifest": str(old / "manifest.jsonl")}
            write_state(root, "dcid", state)
            loaded = last_preparation(root, "dcid")
            selected = Path(loaded["classification_roots"][0])
            self.assertEqual(actual / "classification" / "DCID-7", selected)
            self.assertEqual(str(selected), prepared_training_input(actual, selected)["data"])
            source.write_bytes(b"changed source")
            self.assertIsNone(last_preparation(root, "dcid"))

    def test_only_prepared_source_documents_are_added_to_search(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            folder = root / "iodp362" / "prepared" / "run1"
            (folder / "knowledge").mkdir(parents=True)
            (folder / "report.json").write_text("{}")
            (folder / "candidates.jsonl").write_text("")
            write_state(root, "iodp362", {"preparation": {"output_dir": str(folder),
                        "manifest": str(folder / "candidates.jsonl"), "knowledge_dir": str(folder / "knowledge")}})
            self.assertEqual([folder / "knowledge"], prepared_knowledge_folders(root))

    def test_chat_commands_do_not_hijack_geology_questions(self):
        self.assertTrue(wants_open_datasets("открой учебные материалы"))
        self.assertTrue(wants_open_datasets("добавь датасет"))
        self.assertFalse(wants_open_datasets("Что такое датасет и как устроена фация?"))
        self.assertFalse(wants_open_datasets("Найди фото скважины Р-25"))


class OfflineTrainerTests(unittest.TestCase):
    def test_importing_trainer_does_not_import_gpu_or_download_libraries(self):
        original_import = builtins.__import__
        def cold_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "ultralytics", "matplotlib"}:
                raise AssertionError("Heavy import before explicit training: " + name)
            return original_import(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("_cold_cv_trainer", open_dataset_train.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=cold_import):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

    def test_network_guard_blocks_connections_and_dns_without_actual_network(self):
        with patch.dict(os.environ), patch.object(socket, "create_connection"), patch.object(socket, "getaddrinfo"), \
             patch.object(socket.socket, "connect"), patch.object(socket.socket, "connect_ex"):
            open_dataset_train.install_offline_network_guard()
            for operation in (lambda: socket.create_connection(("example.invalid", 443)),
                              lambda: socket.getaddrinfo("example.invalid", 443),
                              lambda: socket.socket.connect(None, ("127.0.0.1", 443)),
                              lambda: socket.socket.connect_ex(None, ("example.invalid", 443))):
                with self.assertRaisesRegex(ConnectionError, "отключены"):
                    operation()
            self.assertEqual("true", os.environ["YOLO_OFFLINE"])
            self.assertEqual("false", os.environ["YOLO_AUTOINSTALL"])

    def test_font_checks_use_only_bundled_font_and_download_aliases_are_disabled(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            font = root / "fonts" / "ttf" / "DejaVuSans.ttf"
            font.parent.mkdir(parents=True)
            font.write_bytes(b"local font fixture")
            fake = {name: SimpleNamespace() for name in (
                "ultralytics.utils.checks", "ultralytics.data.utils", "ultralytics.utils.plotting",
                "ultralytics.utils.downloads")}
            fake["matplotlib"] = SimpleNamespace(get_data_path=lambda: str(root))
            with patch.object(open_dataset_train.importlib, "import_module", side_effect=lambda name: fake[name]):
                open_dataset_train.configure_offline_ultralytics()
            for name in ("ultralytics.utils.checks", "ultralytics.data.utils", "ultralytics.utils.plotting"):
                self.assertEqual(font, fake[name].check_font("Arial.Unicode.ttf"))
            for name in ("ultralytics.utils.downloads", "ultralytics.data.utils"):
                with self.assertRaises(ConnectionError):
                    fake[name].safe_download(url="https://example.invalid/font.ttf")
            font.unlink()
            self.assertIsNone(fake["ultralytics.data.utils"].check_font("missing.ttf"))

    def test_training_installs_guard_before_dependencies_and_disables_amp(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            for split in ("train", "val"):
                images = root / "images" / split
                images.mkdir(parents=True)
                (images / "fixture.jpg").write_bytes(b"image fixture")
            (root / "summary.json").write_text(json.dumps({
                "task": "segmentation", "readiness": "ready", "source_id": "unibo",
                "classes": [{"id": 0, "name": "external_source_facies"}],
            }), encoding="utf-8")
            (root / "data.yaml").write_text("names: [external_source_facies]\n", encoding="utf-8")
            initial = root / "initial.pt"
            initial.write_bytes(b"local model fixture")
            run = root / "models" / "run"
            (run / "weights").mkdir(parents=True)
            (run / "weights" / "best.pt").write_bytes(b"candidate fixture")
            model = Mock(task="segment", trainer=SimpleNamespace(save_dir=run))
            candidate = Mock(names={0: "external_source_facies"})
            candidate.val.return_value = SimpleNamespace(results_dict={"fixture_metric": 0.5})
            yolo = Mock(side_effect=[model, candidate])
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            events = []
            original_import = builtins.__import__
            def checked_import(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "ultralytics"}:
                    self.assertIn("guard", events)
                return original_import(name, *args, **kwargs)
            arguments = ["open_dataset_train.py", "--prepared", str(root), "--model", str(initial),
                         "--output", str(root / "models"), "--epochs", "1"]
            with patch.object(sys, "argv", arguments), \
                 patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": SimpleNamespace(YOLO=yolo)}), \
                 patch.object(open_dataset_train, "install_offline_network_guard", side_effect=lambda: events.append("guard")), \
                 patch.object(open_dataset_train, "configure_offline_ultralytics", side_effect=lambda: events.append("configured")), \
                 patch("builtins.__import__", side_effect=checked_import):
                open_dataset_train.main()
            self.assertEqual(["guard", "configured"], events)
            self.assertFalse(model.train.call_args.kwargs["amp"])
            self.assertFalse(candidate.val.call_args.kwargs["amp"])
            self.assertFalse(model.train.call_args.kwargs["plots"])
            report = json.loads((run / "model_candidate.json").read_text(encoding="utf-8"))
            self.assertEqual({"0": "external_source_facies"}, report["classes"])
            self.assertFalse(report["auto_activated"])
            self.assertTrue(report["offline_training"])


class DialogTests(unittest.TestCase):
    def test_training_is_disabled_without_validation_and_local_archives_are_visible(self):
        from PySide6.QtWidgets import QApplication, QWidget
        from app.ui.dialogs.open_datasets_dialog import OpenDatasetsDialog
        app = QApplication.instance() or QApplication([])
        main = QWidget()
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            root = Path(temp)
            folder = root / "open_datasets" / "unibo" / "prepared" / "run1"
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text("{}")
            (folder / "manifest.jsonl").write_text("")
            (folder / "data.yaml").write_text("names: [facies]\n")
            write_state(root / "open_datasets", "unibo", {"preparation": {
                "output_dir": str(folder), "manifest": str(folder / "manifest.jsonl"),
                "readiness": "needs_validation",
            }})
            with patch("app.ui.dialogs.open_datasets_dialog.user_data_root", return_value=root):
                dialog = OpenDatasetsDialog(main, source_id="unibo")
                self.assertTrue(dialog.review_button.isEnabled())
                self.assertFalse(dialog.train_button.isEnabled())
                dialog.close()
        main.close()

    def test_cancel_training_terminates_process_and_keeps_cancel_state(self):
        from PySide6.QtCore import QProcess
        from PySide6.QtWidgets import QApplication, QWidget
        from app.ui.dialogs.open_datasets_dialog import OpenDatasetsDialog
        app = QApplication.instance() or QApplication([])
        main = QWidget()
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temp:
            with patch("app.ui.dialogs.open_datasets_dialog.user_data_root", return_value=Path(temp)):
                dialog = OpenDatasetsDialog(main)
            process = Mock()
            process.state.return_value = QProcess.ProcessState.Running
            dialog._process = process
            with patch("app.ui.dialogs.open_datasets_dialog.QTimer.singleShot") as timer:
                dialog._cancel()
            self.assertTrue(dialog._training_cancelled)
            process.terminate.assert_called_once()
            timer.assert_called_once()
            dialog._kill_cancelled_training()
            process.kill.assert_called_once()
            process.state.return_value = QProcess.ProcessState.NotRunning
            dialog.close()
        main.close()


if __name__ == "__main__":
    unittest.main()
