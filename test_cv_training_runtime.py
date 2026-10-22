"""Runtime selection is tested with mocked processes, never local GPU imports."""

from __future__ import annotations

import json
import inspect
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.infrastructure.cv_training_runtime import (
    CvTrainingRuntimeCancelled, CvTrainingRuntimeError,
    resolve_cv_training_python, select_cv_training_runtime,
)


MODULE = "app.infrastructure.cv_training_runtime"


def _result(*, cuda=True, errors=None):
    return {"ok": not errors, "errors": errors or {}, "cuda_available": cuda,
            "cuda_version": "12.6" if cuda else "", "device_name": "NVIDIA test" if cuda else "",
            "versions": {"torch": "2.6.0+cu126", "ultralytics": "8.3", "yaml": "6.0"}}


def _process(result):
    process = Mock(returncode=0)
    process.communicate.return_value = ("package log\n__KERN_CV_RUNTIME__" + json.dumps(result) + "\n", "")
    return process


def _python(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path.resolve()


class CvTrainingRuntimeTests(unittest.TestCase):
    def test_cold_cuda_import_budget_is_bounded_and_configurable(self):
        parameters = inspect.signature(select_cv_training_runtime).parameters
        self.assertEqual(180.0, parameters["timeout_seconds"].default)
        self.assertEqual(300.0, parameters["total_timeout_seconds"].default)
        for value in (0, 301, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_cv_training_runtime(timeout_seconds=value)

    def test_current_python_is_used_only_after_successful_offline_cuda_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = _python(root, "current-python.exe")
            process = _process(_result())
            with patch(MODULE + ".sys.executable", str(python)), patch(MODULE + ".subprocess.Popen", return_value=process) as popen:
                runtime = select_cv_training_runtime(root=root)
            self.assertEqual(python, runtime.python)
            self.assertTrue(runtime.cuda_available)
            self.assertEqual("NVIDIA test", runtime.device_name)
            arguments = popen.call_args
            self.assertEqual([str(python), "-I", "-c"], arguments.args[0][:3])
            self.assertIn("socket.socket.connect = offline", arguments.args[0][3])
            self.assertEqual("true", arguments.kwargs["env"]["YOLO_OFFLINE"])
            self.assertEqual("false", arguments.kwargs["env"]["YOLO_AUTOINSTALL"])
            self.assertEqual(subprocess.DEVNULL, arguments.kwargs["stdin"])
            self.assertNotIn("shell", arguments.kwargs)

    def test_missing_libraries_and_cpu_python_fall_back_to_configured_cuda_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = _python(root, "current.exe")
            local = _python(root, ".venv/Scripts/python.exe")
            vlm = _python(root, "vlm-python.exe")
            processes = [_process(_result(errors={"ultralytics": "ModuleNotFoundError"})),
                         _process(_result(cuda=False)), _process(_result())]
            with patch(MODULE + ".sys.executable", str(current)), patch(MODULE + ".subprocess.Popen", side_effect=processes) as popen:
                result = resolve_cv_training_python({"python": str(vlm)}, root=root)
            self.assertEqual(vlm, result)
            self.assertEqual([str(current), str(local), str(vlm)], [call.args[0][0] for call in popen.call_args_list])

    def test_packaged_application_is_not_started_as_python_and_candidates_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = _python(root, "Kern_Analyzer.exe")
            local = _python(root, ".venv/Scripts/python.exe")
            with patch(MODULE + ".sys.frozen", True, create=True), patch(MODULE + ".sys.executable", str(executable)), \
                 patch(MODULE + ".subprocess.Popen", return_value=_process(_result(cuda=False))) as popen:
                with self.assertRaises(CvTrainingRuntimeError) as caught:
                    select_cv_training_runtime({"python": str(local)}, root=root)
            self.assertEqual(1, popen.call_count)
            self.assertEqual(str(local), popen.call_args.args[0][0])
            self.assertIn("requirements.txt", str(caught.exception))
            self.assertIn("не гарантирует поддержку CUDA", str(caught.exception))
            self.assertIn("CUDA-окружение", str(caught.exception))

    def test_requirements_hint_targets_detected_cuda_python_instead_of_cpu_python(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = _python(root, "cpu-python.exe")
            cuda = _python(root, "cuda-python.exe")
            processes = [_process(_result(cuda=False)),
                         _process(_result(errors={"ultralytics": "ModuleNotFoundError"}))]
            with patch(MODULE + ".sys.executable", str(current)), patch(MODULE + ".subprocess.Popen", side_effect=processes):
                with self.assertRaises(CvTrainingRuntimeError) as caught:
                    select_cv_training_runtime({"python": str(cuda)}, root=root)
            command, = [line for line in str(caught.exception).splitlines() if "-m pip install" in line]
            self.assertIn(str(cuda), command)
            self.assertNotIn(str(current), command)

    def test_malformed_probe_does_not_qualify_as_a_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = _python(root, "current.exe")
            process = Mock(returncode=0)
            process.communicate.return_value = ("__KERN_CV_RUNTIME__{bad json}\n", "")
            with patch(MODULE + ".sys.executable", str(python)), patch(MODULE + ".subprocess.Popen", return_value=process):
                with self.assertRaisesRegex(CvTrainingRuntimeError, "корректный результат"):
                    select_cv_training_runtime(root=root)

    def test_cancel_stops_running_probe_and_never_starts_another(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = _python(root, "current.exe")
            _python(root, ".venv/Scripts/python.exe")
            cancelled = threading.Event()
            process = Mock(returncode=None)
            def communicate(*, timeout):
                if cancelled.is_set():
                    return "", ""
                cancelled.set()
                raise subprocess.TimeoutExpired("probe", timeout)
            process.communicate.side_effect = communicate
            with patch(MODULE + ".sys.executable", str(python)), patch(MODULE + ".subprocess.Popen", return_value=process) as popen:
                with self.assertRaises(CvTrainingRuntimeCancelled):
                    select_cv_training_runtime(root=root, cancel=cancelled)
            process.kill.assert_called_once()
            self.assertEqual(1, popen.call_count)

    def test_hanging_probes_are_killed_within_shared_time_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = _python(root, "current.exe")
            _python(root, ".venv/Scripts/python.exe")
            _python(root, "another-python.exe")
            elapsed = [0.0]
            processes = []
            def start(*args, **kwargs):
                process = Mock(returncode=None)
                killed = [False]
                def communicate(*, timeout):
                    if killed[0]:
                        return "", ""
                    elapsed[0] += timeout
                    raise subprocess.TimeoutExpired("probe", timeout)
                process.communicate.side_effect = communicate
                process.kill.side_effect = lambda: killed.__setitem__(0, True)
                processes.append(process)
                return process
            with patch(MODULE + ".sys.executable", str(python)), \
                 patch(MODULE + ".subprocess.Popen", side_effect=start) as popen, \
                 patch(MODULE + ".time.monotonic", side_effect=lambda: elapsed[0]):
                with self.assertRaises(CvTrainingRuntimeError) as caught:
                    select_cv_training_runtime({"python": str(root / "another-python.exe")}, root=root,
                                               timeout_seconds=1, total_timeout_seconds=1.5)
            self.assertEqual(2, popen.call_count)
            self.assertLessEqual(elapsed[0], 1.500001)
            for process in processes:
                process.kill.assert_called_once()
            self.assertIn("время проверки", str(caught.exception))

    def test_pre_cancelled_selection_never_launches_processes(self):
        event = threading.Event()
        event.set()
        with patch(MODULE + ".subprocess.Popen") as popen:
            with self.assertRaises(CvTrainingRuntimeCancelled):
                select_cv_training_runtime(cancel=event)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
