"""Entry point for Kern Analyzer.

The same portable executable serves both the graphical workplace and the
no-dialog processing routes used by the accompanying BAT shortcuts.  With no
arguments it opens the GUI; with a core-tape command it delegates to the
automatic pipeline and writes results beside the supplied data.
"""

import sys
from pathlib import Path


# Share the trained description-network implementation with the standalone
# studio in source checkouts. PyInstaller includes the package in its archive.
studio_source = Path(__file__).resolve().parent / "excel_photo_model_studio" / "src"
if studio_source.is_dir() and str(studio_source) not in sys.path:
    sys.path.insert(0, str(studio_source))


def _enable_command_console() -> None:
    """Attach a console for BAT-driven routes in a windowed PyInstaller EXE.

    ``console=False`` is right for the normal graphical start.  It leaves
    ``sys.stdout`` and ``sys.stderr`` unavailable, however, which breaks
    argparse and status messages when the very same executable is started by
    a command shortcut.  Reuse the parent BAT console when possible; if the
    command was launched directly, create a small one for readable progress.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            if not kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
                kernel32.AllocConsole()
            console = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
            sys.stdout = console
            sys.stderr = console
            return
        except OSError:
            pass
    # Last-resort fallback keeps command routes functional even if a console
    # cannot be created (for example under a restricted launcher).
    sink = open("NUL" if sys.platform == "win32" else "/dev/null", "w", encoding="utf-8")
    sys.stdout = sink
    sys.stderr = sink


def main() -> int:
    if sys.argv[1:] == ["--vision-worker"]:
        from app.infrastructure.local_vision_runner import main as run_vision
        run_vision()
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--hub-sync-job":
        from app.infrastructure.model_hub_worker import main as run_hub_job
        return run_hub_job(sys.argv[2])
    if len(sys.argv) > 1 and sys.argv[1] == "--portable-smoke-test":
        _enable_command_console()
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import cv2  # noqa: F401 - verifies bundled runtime
        import openpyxl  # noqa: F401 - verifies Excel import runtime
        import torch  # noqa: F401 - verifies model runtime
        from PySide6.QtWidgets import QApplication
        from app.domain.facies_catalog import FACIES_MODEL_CLASSES
        from app.ui.windows.main_window import MainWindow
        from app.ui.dialogs.open_datasets_dialog import OpenDatasetsDialog

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        datasets = OpenDatasetsDialog(window)
        if datasets.table.rowCount() != 6:
            raise RuntimeError("Portable open dataset catalog is incomplete")
        datasets.close()
        import json
        from PySide6.QtCore import QProcess
        from app.infrastructure.local_vision import default_model_folder
        from app.runtime_paths import bundled_root

        worker = QProcess()
        command = ["--vision-worker"]
        if not getattr(sys, "frozen", False):
            command.insert(0, str(Path(__file__).resolve()))
        worker.start(sys.executable, command)
        if not worker.waitForStarted(5000):
            raise RuntimeError("Portable Vision worker could not start")
        requests = [{"id": "smoke", "operation": "chat", "question": "hello"}]
        model = default_model_folder()
        if (model / "model.pt").is_file():
            requests.append({
                "id": "photo", "operation": "describe", "base": str(model),
                "image": str(bundled_root() / "app/resources/lithology_examples/sandstone.jpg"),
            })
        worker.write("".join(json.dumps(row) + "\n" for row in requests).encode("utf-8"))
        worker.closeWriteChannel()
        if not worker.waitForFinished(120000):
            worker.kill()
            worker.waitForFinished(5000)
            raise RuntimeError("Portable Vision worker timed out")
        replies = [json.loads(line) for line in bytes(worker.readAllStandardOutput()).decode("utf-8").splitlines()]
        if (worker.exitCode() != 0 or [row.get("id") for row in replies] != [row["id"] for row in requests]
                or not all(row.get("ok") for row in replies)):
            error = bytes(worker.readAllStandardError()).decode("utf-8", errors="replace")
            raise RuntimeError(f"Portable Vision worker failed: {replies}\n{error}")
        if len(requests) > 1 and replies[-1].get("model_info", {}).get("generative") is not False:
            raise RuntimeError("Portable classifier did not report the Vision engine")
        window.close()
        app.quit()
        if len(FACIES_MODEL_CLASSES) != 92:
            raise RuntimeError("Portable facies registry is incomplete")
        print("Kern Analyzer portable smoke test: OK")
        return 0
    if len(sys.argv) > 1:
        # Keep the command-line workflow inside the full GUI distribution so
        # that v1.5 never depends on a second, lightweight executable.
        _enable_command_console()
        from build_core_tape import main as core_tape_main

        return core_tape_main()
    # Keep Qt out of command-line modes.  This makes the automatic agents
    # usable even on a machine where the graphical Qt runtime cannot start.
    from PySide6.QtWidgets import QApplication
    from app.ui.windows.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Kern Analyzer")
    window = MainWindow()
    window.showMaximized()
    from PySide6.QtCore import QTimer
    QTimer.singleShot(0, window.prepare_local_runtime)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
